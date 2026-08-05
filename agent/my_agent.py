from __future__ import annotations

# =====================================================================
# vLLM-driven ARC-AGI-3 submission agent
# The policy is served locally through vLLM's OpenAI-compatible API.
# =====================================================================
import base64
import difflib
import hashlib
import io
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from typing import Any

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI
from PIL import Image

from agents.agent import Agent

logger = logging.getLogger(__name__)

# All MyAgent instances share one submission budget.  Swarm creates one agent
# per game concurrently, so an instance-local timer would allow the aggregate
# run to exceed Kaggle's wall-clock limit.
_SUBMISSION_STARTED_AT = time.monotonic()


class MyAgent(Agent):
    """vLLM-powered ARC agent that emits one JSON action per step."""

    name = os.getenv("ARC_AGENT_NAME", "forge_v37_gemma31b_multicandidate_arbiter")

    MODEL_TO_GAME_ACTION = {
        "up": "ACTION1",
        "down": "ACTION2",
        "left": "ACTION3",
        "right": "ACTION4",
        "spacebar": "ACTION5",
        "click": "ACTION6",
        "undo": "ACTION7",
        "reset": "RESET",
    }
    GAME_TO_MODEL_ACTION = {
        game_name: model_name for model_name, game_name in MODEL_TO_GAME_ACTION.items()
    }
    _DEFAULT_MAX_ACTIONS = 200
    # Submission safety limits, matching the official GPT-OSS template style.
    # Swarm runs one thread per game; each thread must finish so the scorecard can close.
    GAME_TIME_LIMIT_S = 8 * 60 * 60
    FIRST_ACTION_DEADLINE_S = 14 * 60
    LLM_REQUEST_TIMEOUT_S = 400
    GLOBAL_TIME_LIMIT_SECONDS = 9 * 60 * 60
    GLOBAL_SHUTDOWN_RESERVE_SECONDS = 20 * 60
    MODEL_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
    MAX_HISTORY = 12
    MAX_SIGNIFICANT_EVENTS = 50
    MAX_FRAME_MEMORY = 11
    ACTION_CONTEXT_FRAMES = 4
    REFLECTION_INTERVAL = 10
    MAX_REFLECTION_CHARS = 1800
    MAX_SHARED_MECHANISM_CHARS = 4000
    MAX_SHARED_MECHANISM_PROMPT_CHARS = 500
    MAX_PLAN_ACTIONS = 4
    FRAME_BORDER_IGNORE = 3
    MAX_NEW_TOKENS = 1024
    REPAIR_MAX_NEW_TOKENS = 256
    REFLECTION_MAX_NEW_TOKENS = 10000
    FRAME_IMAGE_SCALE = 8
    ACTION_CANDIDATES = 3
    ARBITER_MAX_NEW_TOKENS = 512
    DEFAULT_TRACE_PATH = "/kaggle/working/llm_inference_trace.jsonl"
    VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
    VLLM_SERVED_MODEL_NAME = "vllm-model"
    VLLM_LOG_PATH = "/kaggle/working/vllm_server.log"
    ARC_PALETTE = [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (57, 204, 204),
        (177, 13, 201),
        (1, 255, 112),
        (133, 20, 75),
        (61, 153, 112),
        (221, 221, 221),
    ]
    LABEL_GLYPHS = {
        " ": ["000", "000", "000", "000", "000"],
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
        "E": ["111", "100", "111", "100", "111"],
        "P": ["111", "101", "111", "100", "100"],
        "S": ["111", "100", "111", "001", "111"],
        "T": ["111", "010", "010", "010", "010"],
    }
    _client: OpenAI | None = None
    _served_model: str | None = None
    _server_process: subprocess.Popen[bytes] | None = None
    _server_log: Any = None
    _server_lock = threading.Lock()
    # Swarm runs one thread per game concurrently against the same process,
    # so a cross-game shared file needs its own lock independent of
    # _server_lock (which guards vLLM client/process state, not files).
    _shared_mechanism_lock = threading.Lock()
    _vllm_startup_error: str | None = None
    _startup_attempts: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed_material = ":".join(
            [
                os.getenv("ARC_AGENT_NAME", self.name),
                str(self.game_id),
                os.getenv("AGENT_RANDOM_SEED", "0"),
            ]
        )
        seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:12], 16)
        self.rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []
        self.frame_memory: list[dict[str, Any]] = []
        self.pending_actions: list[dict[str, Any]] = []
        self.last_plan_summary = ""
        self.reflection_buffer: list[dict[str, Any]] = []
        # Unlike reflection_buffer (drained into prose every REFLECTION_INTERVAL
        # steps and then discarded), this keeps a compact record of only the
        # rare "something actually happened" steps for the whole game, so a
        # rule learned in level 1 isn't fully gone by level 5.
        self.significant_events: list[dict[str, Any]] = []
        self.reflection_memory_path = self._reflection_memory_path()
        self.reflection_memory = self._load_reflection_memory()
        # Cross-game hints only -- never a hard action filter. Loaded once
        # here and refreshed at each reflection cycle (not every action) so
        # a mechanic another concurrently-running game just confirmed can
        # still reach this game while it's playing, without adding a file
        # read to the hot per-action path.
        self.shared_mechanisms = self._load_shared_mechanisms()
        self.reflections_completed = 0
        self.reflection_failures = 0
        self.current_level_number = 1
        self.failed_state_actions: dict[str, set[str]] = {}
        # Exact frame hashes never repeat in games with an animation, timer or
        # step counter, so also tally failures against a coarse object-layout
        # key. Counts, not a set: one coincidence should not ban an action.
        self.failed_abstract_actions: dict[str, dict[str, int]] = {}
        self._game_started_monotonic = time.monotonic()
        self._deadline_hit = False

    @staticmethod
    def _memory_base_dir() -> str:
        # sys.platform check first: Kaggle's runtime is always Linux, so this
        # never changes real submission behaviour. Without it, a leading "/"
        # is drive-relative on Windows (no path separator translation the way
        # Git Bash gives you), so os.path.isdir("/kaggle/working") silently
        # matches an unrelated pre-existing C:\kaggle\working directory on a
        # dev machine and redirects memory there instead of next to the repo.
        default_dir = (
            "/kaggle/working/agent_memory"
            if sys.platform != "win32" and os.path.isdir("/kaggle/working")
            else os.path.join(os.getcwd(), "agent_memory")
        )
        return os.getenv("LLM_MEMORY_DIR", default_dir)

    def _reflection_memory_path(self) -> str:
        safe_game_id = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in self.game_id
        )
        return os.path.join(self._memory_base_dir(), f"{safe_game_id}.md")

    @classmethod
    def _shared_mechanism_memory_path(cls) -> str:
        return os.path.join(cls._memory_base_dir(), "_shared_mechanisms.md")

    def _load_reflection_memory(self) -> str:
        try:
            with open(self.reflection_memory_path, "r", encoding="utf-8") as memory_file:
                memory = memory_file.read().strip()
            if memory:
                return memory[: self.MAX_REFLECTION_CHARS]
        except OSError:
            pass
        return "# Agent Memory\n\nNo reflection has been completed yet."

    def _shared_mechanism_memory_enabled(self) -> bool:
        return self._env_flag("LLM_SHARED_MECHANISM_MEMORY", "0")

    def _load_shared_mechanisms(self) -> str:
        if not self._shared_mechanism_memory_enabled():
            return ""
        try:
            with open(
                self._shared_mechanism_memory_path(), "r", encoding="utf-8"
            ) as memory_file:
                return memory_file.read().strip()[: self.MAX_SHARED_MECHANISM_CHARS]
        except OSError:
            return ""

    @staticmethod
    def _looks_like_generalizable_rule(rule_text: str, game_id: str) -> bool:
        """Reject rules that look tied to this specific game/level.

        Cheap heuristic, not a proof: cross-game notes are hints the prompt
        already labels as unverified, so a false positive here just adds one
        unhelpful line rather than actively misleading the model the way a
        wrong entry in the hard action-failure filters would.
        """
        lowered = rule_text.lower()
        if game_id.lower() in lowered:
            return False
        return not any(
            marker in lowered for marker in ("level ", "this game", "this level")
        )

    def _extract_confirmed_rules(self, reflection_text: str) -> list[str]:
        rules: list[str] = []
        for raw_line in reflection_text.splitlines():
            line = raw_line.strip().lstrip("-*").strip()
            if "[CONFIRMED]" in line and self._looks_like_generalizable_rule(
                line, self.game_id
            ):
                rules.append(line)
        return rules

    @staticmethod
    def _is_similar_to_existing(candidate: str, existing_text: str, threshold: float = 0.6) -> bool:
        candidate_norm = candidate.strip().lower()
        for existing_line in existing_text.splitlines():
            existing_norm = existing_line.strip().lower()
            if not existing_norm:
                continue
            if difflib.SequenceMatcher(None, candidate_norm, existing_norm).ratio() >= threshold:
                return True
        return False

    def _update_shared_mechanisms(self) -> None:
        if not self._shared_mechanism_memory_enabled():
            return
        confirmed_rules = self._extract_confirmed_rules(self.reflection_memory)
        if not confirmed_rules:
            return
        path = self._shared_mechanism_memory_path()
        try:
            with self._shared_mechanism_lock:
                try:
                    with open(path, "r", encoding="utf-8") as memory_file:
                        existing = memory_file.read()
                except OSError:
                    existing = ""
                new_lines = [
                    rule
                    for rule in confirmed_rules
                    if not self._is_similar_to_existing(rule, existing)
                ]
                if not new_lines:
                    return
                lines = [line for line in existing.splitlines() if line.strip()]
                lines.extend(new_lines)
                # FIFO: drop the oldest lines first once the total size is
                # over budget, so the file can't grow without bound across a
                # long multi-game submission.
                while lines and sum(len(line) + 1 for line in lines) > self.MAX_SHARED_MECHANISM_CHARS:
                    lines.pop(0)
                combined = "\n".join(lines)
                memory_dir = os.path.dirname(path)
                if memory_dir:
                    os.makedirs(memory_dir, exist_ok=True)
                temp_path = path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as memory_file:
                    memory_file.write(combined + "\n")
                os.replace(temp_path, path)
                self.shared_mechanisms = combined
        except OSError as exc:
            logger.warning("Failed to update shared mechanism memory: %s", exc)

    @classmethod
    def _global_deadline(cls) -> float:
        try:
            limit = float(
                os.getenv(
                    "AGENT_GLOBAL_TIME_LIMIT_SECONDS",
                    str(cls.GLOBAL_TIME_LIMIT_SECONDS),
                )
            )
        except ValueError:
            limit = float(cls.GLOBAL_TIME_LIMIT_SECONDS)
        try:
            reserve = float(
                os.getenv(
                    "AGENT_GLOBAL_SHUTDOWN_RESERVE_SECONDS",
                    str(cls.GLOBAL_SHUTDOWN_RESERVE_SECONDS),
                )
            )
        except ValueError:
            reserve = float(cls.GLOBAL_SHUTDOWN_RESERVE_SECONDS)
        return _SUBMISSION_STARTED_AT + max(0.0, limit - max(0.0, reserve))

    @classmethod
    def _remaining_global_seconds(cls) -> float:
        return max(0.0, cls._global_deadline() - time.monotonic())

    @classmethod
    def _global_time_budget_fraction_remaining(cls) -> float:
        """1.0 at submission start, 0.0 at the global deadline.

        Reuses _global_deadline (already limit-minus-reserve) so this stays
        consistent with is_done's own budget accounting instead of
        recomputing the window a second way.
        """
        total_window = cls._global_deadline() - _SUBMISSION_STARTED_AT
        if total_window <= 0:
            return 0.0
        return max(0.0, min(1.0, cls._remaining_global_seconds() / total_window))

    @classmethod
    def _load_vllm_once(cls) -> None:
        if cls._client is not None and cls._served_model is not None:
            return

        with cls._server_lock:
            if cls._client is not None and cls._served_model is not None:
                return

            port = os.getenv("VLLM_PORT", "8000")
            default_base_url = f"http://127.0.0.1:{port}/v1"
            base_url = os.getenv("VLLM_BASE_URL", default_base_url).rstrip("/")
            remaining = cls._remaining_global_seconds()
            if remaining <= 0:
                raise TimeoutError("Global submission time budget exhausted before vLLM startup")
            request_timeout = min(
                float(os.getenv("VLLM_REQUEST_TIMEOUT", "1200")), remaining
            )
            client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("VLLM_API_KEY", "local-server-key"),
                timeout=max(1.0, request_timeout),
                max_retries=0,
            )

            try:
                models = client.models.list()
            except Exception:
                if os.getenv("VLLM_START_SERVER", "1").lower() in {"0", "false", "no"}:
                    raise RuntimeError(f"No vLLM server is reachable at {base_url}")
                cls._start_vllm_server()
                models = cls._wait_for_vllm(client)

            if not models.data:
                raise RuntimeError("vLLM reported no served models")
            requested_model = os.getenv(
                "VLLM_SERVED_MODEL_NAME", cls.VLLM_SERVED_MODEL_NAME
            )
            model_ids = {item.id for item in models.data}
            cls._served_model = (
                requested_model if requested_model in model_ids else models.data[0].id
            )
            cls._client = client
            logger.info("vLLM ready at %s with model %s", base_url, cls._served_model)

    @classmethod
    def _teardown_server(cls) -> None:
        """Kill the vLLM server and its whole process group.

        The engine runs worker children; terminating only the parent leaves
        them holding GPU memory, which makes the next startup OOM.
        """
        process = cls._server_process
        if process is not None and process.poll() is None:
            for signal_number in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(process.pid), signal_number)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except Exception:
                        pass
                try:
                    process.wait(timeout=30)
                    break
                except Exception:
                    continue
        cls._server_process = None
        if cls._server_log is not None:
            try:
                cls._server_log.close()
            except Exception:
                pass
            cls._server_log = None
        cls._client = None
        cls._served_model = None

    @classmethod
    def _degrade_startup_settings(cls, attempt: int) -> None:
        """Back off memory-hungry settings before retrying startup."""
        utilizations = ["0.94", "0.88", "0.80"]
        model_lengths = ["32768", "16384", "8192"]
        index = min(attempt, len(utilizations) - 1)
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = utilizations[index]
        os.environ["VLLM_MAX_MODEL_LEN"] = model_lengths[index]
        logger.warning(
            "Retrying vLLM startup with gpu_memory_utilization=%s max_model_len=%s",
            utilizations[index],
            model_lengths[index],
        )

    @classmethod
    def _startup_attempt_limit(cls) -> int:
        try:
            return max(1, int(os.getenv("VLLM_STARTUP_ATTEMPTS", "3")))
        except ValueError:
            return 3

    @classmethod
    def _ensure_vllm_available(cls) -> None:
        if cls._vllm_startup_error is not None:
            raise RuntimeError(
                f"vLLM disabled after startup failure: {cls._vllm_startup_error}"
            )
        try:
            cls._load_vllm_once()
        except Exception as exc:
            # Latching on the first failure costs the entire remaining budget,
            # so retry a bounded number of times with degraded settings before
            # giving up on the model for good.
            with cls._server_lock:
                cls._startup_attempts += 1
                attempts = cls._startup_attempts
                cls._teardown_server()
                exhausted = (
                    attempts >= cls._startup_attempt_limit()
                    or cls._remaining_global_seconds() <= 0
                )
                if exhausted:
                    cls._vllm_startup_error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "vLLM startup failed %s times; disabling model: %s",
                        attempts,
                        exc,
                    )
                else:
                    cls._degrade_startup_settings(attempts)
            raise

    @classmethod
    def _start_vllm_server(cls) -> None:
        if cls._server_process is not None and cls._server_process.poll() is None:
            return

        cls._configure_cuda_library_path()

        model_path = os.getenv("VLLM_MODEL_PATH", cls.MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"vLLM model path not found: {model_path}. Attach the Kaggle model asset "
                "or set VLLM_MODEL_PATH."
            )

        served_name = os.getenv("VLLM_SERVED_MODEL_NAME", cls.VLLM_SERVED_MODEL_NAME)
        port = os.getenv("VLLM_PORT", "8000")
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--served-model-name",
            served_name,
            "--tensor-parallel-size",
            os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"),
            "--max-num-seqs",
            os.getenv("VLLM_MAX_NUM_SEQS", "20"),
            "--gpu-memory-utilization",
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.94"),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--dtype",
            os.getenv("VLLM_DTYPE", "auto"),
            "--max-model-len",
            os.getenv("VLLM_MAX_MODEL_LEN", "32768"),
            "--enable-prefix-caching",
            "--trust-remote-code",
        ]
        limit_mm_per_prompt = os.getenv("VLLM_LIMIT_MM_PER_PROMPT", "").strip()
        if limit_mm_per_prompt:
            command.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])
        quantization = os.getenv("VLLM_QUANTIZATION", "").strip()
        if quantization:
            command.extend(["--quantization", quantization])
        generation_config = os.getenv("VLLM_GENERATION_CONFIG", "").strip()
        if generation_config:
            command.extend(["--generation-config", generation_config])
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
        cls._server_log = open(log_path, "wb", buffering=0)
        logger.info("Starting vLLM server for %s; log: %s", model_path, log_path)
        cls._server_process = subprocess.Popen(
            command,
            stdout=cls._server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @classmethod
    def _configure_cuda_library_path(cls) -> None:
        paths: list[str] = []
        try:
            import site

            for base in site.getsitepackages():
                nvidia_root = os.path.join(base, "nvidia")
                if os.path.isdir(nvidia_root):
                    for root, dirs, _files in os.walk(nvidia_root):
                        if os.path.basename(root) in {"lib", "lib64"}:
                            paths.append(root)
                        dirs[:] = [name for name in dirs if name not in {"__pycache__"}]
                torch_lib = os.path.join(base, "torch", "lib")
                if os.path.isdir(torch_lib):
                    paths.append(torch_lib)
        except Exception as exc:
            logger.warning("Failed to discover CUDA wheel library paths: %s", exc)
        existing = [
            item
            for item in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
            if item
        ]
        unique: list[str] = []
        seen = set()
        for path in paths + existing:
            if path and path not in seen and os.path.isdir(path):
                unique.append(path)
                seen.add(path)
        if unique:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(unique)

    @classmethod
    def _wait_for_vllm(cls, client: OpenAI) -> Any:
        timeout = min(
            float(os.getenv("VLLM_STARTUP_TIMEOUT", "1000")),
            cls._remaining_global_seconds(),
        )
        if timeout <= 0:
            raise TimeoutError("Global submission time budget exhausted during vLLM startup")
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if cls._server_process is not None and cls._server_process.poll() is not None:
                log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
                log_tail = cls._read_log_tail(log_path)
                raise RuntimeError(
                    f"vLLM server exited with code {cls._server_process.returncode}; "
                    f"see {log_path}\nLast server log lines:\n{log_tail}"
                )
            try:
                return client.models.list()
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
        log_tail = cls._read_log_tail(log_path)
        raise RuntimeError(
            f"vLLM server did not become ready within {timeout:.0f}s: {last_error}\n"
            f"Last server log lines:\n{log_tail}"
        )

    @staticmethod
    def _read_log_tail(path: str, max_bytes: int = 12000) -> str:
        try:
            with open(path, "rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - max_bytes))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError as exc:
            return f"Unable to read vLLM log: {exc}"

    @property
    def MAX_ACTIONS(self) -> int:
        try:
            return max(1, int(os.getenv("AGENT_MAX_ACTIONS", str(self._DEFAULT_MAX_ACTIONS))))
        except ValueError:
            return self._DEFAULT_MAX_ACTIONS

    @property
    def game_elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self._game_started_monotonic)

    @property
    def game_time_remaining_s(self) -> float:
        limit = float(os.getenv("GAME_TIME_LIMIT_S", str(self.GAME_TIME_LIMIT_S)))
        if limit <= 0:
            return self._remaining_global_seconds()
        return max(0.0, min(limit - self.game_elapsed_s, self._remaining_global_seconds()))

    def _mark_deadline_hit(self, reason: str) -> None:
        if not self._deadline_hit:
            logger.info(
                "%s for %s after %s actions and %.2fs",
                reason,
                self.game_id,
                self.action_counter,
                self.game_elapsed_s,
            )
        self._deadline_hit = True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self._remaining_global_seconds() <= 0:
            self._mark_deadline_hit("Global submission time budget exhausted")
            return True
        if self._deadline_hit:
            return True
        # Always allow the first RESET action; the gateway needs early activity.
        if self.action_counter == 0:
            return False
        if self.game_time_remaining_s <= 0:
            self._mark_deadline_hit("Per-game time limit reached")
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        prompt = ""
        response_text = ""
        timings: dict[str, float] = {}
        turn_start = time.perf_counter()
        try:
            if self._remaining_global_seconds() <= 0:
                action = GameAction.RESET
                action.reasoning = "Global submission time budget exhausted."
                return action
            if self.action_counter == 0:
                startup_elapsed_s = time.monotonic() - _SUBMISSION_STARTED_AT
                first_action_deadline_s = float(
                    os.getenv("FIRST_ACTION_DEADLINE_S", str(self.FIRST_ACTION_DEADLINE_S))
                )
                if startup_elapsed_s > first_action_deadline_s:
                    logger.warning(
                        "First action for %s selected after %.2fs, past %.2fs target",
                        self.game_id,
                        startup_elapsed_s,
                        first_action_deadline_s,
                    )
                action = GameAction.RESET
                action.reasoning = "Initial RESET before slow model startup."
                return action
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.pending_actions = []
                action = GameAction.RESET
                action.reasoning = "Environment requires RESET before play."
                return action
            if self.game_time_remaining_s <= 0:
                self._mark_deadline_hit("Per-game time limit reached before LLM action")
                return self._fallback_action(latest_frame, "Per-game time limit reached.")

            # The gateway receives RESET before potentially slow model startup.
            self._ensure_vllm_available()

            stage_start = time.perf_counter()
            self._observe_frame(latest_frame)
            timings["observe_frame"] = time.perf_counter() - stage_start

            reflection_interval = self._reflection_interval()
            if reflection_interval and len(self.reflection_buffer) >= reflection_interval:
                stage_start = time.perf_counter()
                self._run_reflection(latest_frame)
                timings["reflection"] = time.perf_counter() - stage_start

            if self.pending_actions:
                action = self._dequeue_action(latest_frame)
                timings["total_choose_action"] = time.perf_counter() - turn_start
                logger.info(
                    "Agent timing step=%s dequeued_action=%s remaining=%s total_choose_action=%.3fs",
                    self.action_counter,
                    action.name,
                    len(self.pending_actions),
                    timings["total_choose_action"],
                )
                return action

            stage_start = time.perf_counter()
            prompt = self._build_prompt(frames, latest_frame)
            timings["build_prompt"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            frame_images = self._build_context_images(latest_frame.frame)
            timings["build_context_images"] = time.perf_counter() - stage_start
            try:
                stage_start = time.perf_counter()
                response_text, parsed = self._generate_action_response(
                    prompt,
                    frame_images,
                    latest_frame,
                )
                timings["generate_response"] = time.perf_counter() - stage_start
            except Exception as exc:
                self._write_llm_trace(latest_frame, prompt, response_text, context_images=frame_images, error=repr(exc))
                raise

            if parsed is None:
                raise ValueError("Model loop produced no JSON payload")
            if not ("actions" in parsed or "action" in parsed):
                exc = ValueError(f"Model did not finish with an action payload: {parsed}")
                repair_prompt = self._build_json_repair_prompt(prompt, response_text)
                try:
                    stage_start = time.perf_counter()
                    repair_text = self._generate_response(
                        repair_prompt,
                        frame_images,
                        enable_thinking=False,
                        max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
                    )
                    timings["repair_generate_response"] = time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    parsed = self._extract_action_json(repair_text)
                    timings["repair_extract_json"] = time.perf_counter() - stage_start
                    response_text = response_text + "\n\nJSON_REPAIR_OUTPUT:\n" + repair_text
                except Exception:
                    self._write_llm_trace(
                        latest_frame,
                        prompt,
                        response_text,
                        context_images=frame_images,
                        error=repr(exc),
                    )
                    raise exc

            stage_start = time.perf_counter()
            planned_actions = self._normalize_action_specs(parsed, latest_frame)
            if not planned_actions:
                logger.warning("Model returned no usable actions, using ordered fallback: %s", parsed)
                action = self._fallback_action(latest_frame, "Model returned no usable actions.")
                self._write_llm_trace(
                    latest_frame,
                    prompt,
                    response_text,
                    parsed=parsed,
                    chosen_action=action,
                    context_images=frame_images,
                    error="empty_or_unusable_action_plan",
                )
                self._remember_step(latest_frame, action, response_text, parsed)
                timings["plan_to_action"] = time.perf_counter() - stage_start
                timings["total_choose_action"] = time.perf_counter() - turn_start
                self._log_timing(latest_frame, frame_images, timings)
                return action
            self.pending_actions = planned_actions
            self.last_plan_summary = str(parsed.get("plan_summary", "")).strip()
            action = self._dequeue_action(
                latest_frame,
                {
                    "raw_plan": parsed,
                    "plan_length": len(planned_actions),
                    "plan_summary": self.last_plan_summary,
                },
                remember=False,
            )
            timings["plan_to_action"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            self._write_llm_trace(
                latest_frame,
                prompt,
                response_text,
                parsed=parsed,
                chosen_action=action,
                context_images=frame_images,
            )
            timings["write_trace"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            self._remember_step(latest_frame, action, response_text, parsed)
            timings["remember_step"] = time.perf_counter() - stage_start
            timings["total_choose_action"] = time.perf_counter() - turn_start
            self._log_timing(latest_frame, frame_images, timings)
            return action
        except Exception as exc:
            logger.warning("vLLM action generation failed: %s", exc)
            traceback.print_exc()
            try:
                action = self._fallback_action(latest_frame, f"vLLM failure: {exc}")
                if not self.history or self.history[-1].get("step") != self.action_counter:
                    self._remember_step(
                        latest_frame,
                        action,
                        response_text or "FALLBACK_AFTER_VLLM_FAILURE",
                        {
                            "reasoning": "Fallback after model or JSON failure.",
                            "plan_summary": f"Fallback action after error: {exc}",
                        },
                    )
            except Exception as fallback_exc:
                # _fallback_action is the last line of defense; if it also
                # raises (malformed frame, corrupted internal bookkeeping),
                # an uncaught exception here takes the whole submission down
                # instead of just this one step. RESET bypasses availability
                # checks entirely (see is_done/choose_action's action_counter
                # == 0 handling), so it's the one action every state accepts.
                logger.error(
                    "Fallback action generation also failed for %s: %s",
                    self.game_id,
                    fallback_exc,
                )
                action = GameAction.RESET
                action.reasoning = {
                    "fallback": True,
                    "reason": f"vLLM failure ({exc}) and fallback failure ({fallback_exc})",
                }
            return action

    def _build_prompt(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        available_actions = self._available_model_action_names(latest_frame)
        recent_history = json.dumps(self._prompt_history()[-4:], ensure_ascii=True)
        thinking_directive = "/think" if self._action_thinking_enabled() else "/no_think"
        example_action: dict[str, Any] = {"name": available_actions[0]}
        if "click" in available_actions:
            example_action = {"name": "click", "x": 12, "y": 34}
        example_payload: dict[str, Any] = {
            "board_change_assessment": "central-board evidence from the latest transition",
            "plan_summary": "test one rule or pursue the current subgoal",
            "actions": [example_action],
        }
        if self._confidence_prompt_enabled():
            example_payload = {
                "confidence": 0.72,
                "board_change_assessment": "central-board evidence from the latest transition",
                "controllable_object": "object or cursor inferred from transitions",
                "goal_hypothesis": "what must change to advance the level",
                "plan_summary": "test one rule or pursue the current subgoal",
                "actions": [example_action],
            }
        output_example = json.dumps(example_payload, ensure_ascii=True)
        ineffective_actions = self._ineffective_actions_for_current_state(latest_frame)
        legal_action_instructions = self._legal_action_instructions(available_actions)
        frame_descriptor_block = ""
        if self._include_frame_descriptor():
            frame_descriptor = json.dumps(
                self._frame_descriptor(latest_frame.frame),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            frame_descriptor_block = f"\n\nCurrent frame descriptor:\n{frame_descriptor}"
        shared_mechanism_block = ""
        if self.shared_mechanisms:
            shared_mechanism_block = (
                "\n\nCross-game notes (unverified in this game, treat as hints,"
                f" not facts):\n{self.shared_mechanisms[: self.MAX_SHARED_MECHANISM_PROMPT_CHARS]}"
            )
        confidence_instruction = ""
        if self._confidence_prompt_enabled():
            confidence_instruction = (
                "\nInclude confidence from 0.0 to 1.0. If confidence is below 0.55,"
                "\nchoose a reversible diagnostic action instead of a long plan."
            )

        return textwrap.dedent(
            f"""
            You are the action agent for an interactive ARC-AGI-3 visual game.
            The images are chronological; the last is current. Red STEP labels are added
            chronology, not game UI. Ignore the outer {self._border_ignore_pixels()} pixels
            when judging progress. Trust numeric transitions over visual guesses.

            Infer the controllable object, causal action effects, and current objective.
            Prefer purposeful new states. A repeated state is not progress. Do not invent
            counters, bars, or goals without evidence.

            Legal actions for this exact state: {available_actions}
            Action format rules for this state only:
            {legal_action_instructions}
            Do not output any action name outside Legal actions for this exact state.
            Ineffective in this exact state: {ineffective_actions}

            Reflection memory (authoritative but revisable):
            {self.reflection_memory}{shared_mechanism_block}

            Recent transitions:
            {recent_history}{frame_descriptor_block}

            Return exactly one JSON object, no tools or markdown. Include 1 to
            {self._max_plan_actions()} actions; use one exploratory action if uncertain.{confidence_instruction}
            Example: {output_example}
            {thinking_directive}
            """
        ).strip()

    def _env_flag(self, name: str, default: str) -> bool:
        value = os.getenv(name, default).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _adaptive_max_new_tokens(self, base_budget: int) -> int:
        """Scale the main action-generation token budget by remaining time.

        Off by default (LLM_ADAPTIVE_TOKEN_BUDGET=0) so it never changes
        behaviour unless explicitly enabled. _action_candidate_count already
        establishes the precedent of cutting an expensive feature once the
        budget gets tight (dropping to 1 candidate under
        LLM_CANDIDATE_MIN_SECONDS) -- this generalises the same idea to
        token count instead of candidate count.
        """
        if not self._env_flag("LLM_ADAPTIVE_TOKEN_BUDGET", "0"):
            return base_budget
        fraction_remaining = self._global_time_budget_fraction_remaining()
        high_fraction, low_fraction = 0.50, 0.15
        low_scale = 0.60
        if fraction_remaining >= high_fraction:
            scale = 1.0
        elif fraction_remaining <= low_fraction:
            scale = low_scale
        else:
            span = high_fraction - low_fraction
            t = (fraction_remaining - low_fraction) / span
            scale = low_scale + t * (1.0 - low_scale)
        # Never scale below REPAIR_MAX_NEW_TOKENS: a budget that's too small
        # to hold a valid JSON action truncates output on every call and
        # feeds the repair loop instead of saving time.
        return max(self.REPAIR_MAX_NEW_TOKENS, int(base_budget * scale))

    def _include_frame_descriptor(self) -> bool:
        return self._env_flag("LLM_INCLUDE_FRAME_DESCRIPTOR", "1")

    def _confidence_prompt_enabled(self) -> bool:
        return self._env_flag("LLM_CONFIDENCE_PROMPT", "1")

    def _reflection_interval(self) -> int:
        try:
            return max(0, int(os.getenv("LLM_REFLECTION_INTERVAL", str(self.REFLECTION_INTERVAL))))
        except ValueError:
            return self.REFLECTION_INTERVAL

    def _reflection_max_new_tokens(self) -> int:
        try:
            return max(
                64,
                int(os.getenv("LLM_REFLECTION_MAX_NEW_TOKENS", str(self.REFLECTION_MAX_NEW_TOKENS))),
            )
        except ValueError:
            return self.REFLECTION_MAX_NEW_TOKENS

    def _mm_image_limit(self) -> int:
        """Max images the served model accepts per request.

        vLLM is launched with --limit-mm-per-prompt and rejects any request
        carrying more images than that, so every call site must clamp to it.
        """
        raw = os.getenv("VLLM_LIMIT_MM_PER_PROMPT", "").strip()
        if not raw:
            return self.MAX_FRAME_MEMORY
        try:
            limit = int(json.loads(raw).get("image", self.MAX_FRAME_MEMORY))
        except (ValueError, TypeError, AttributeError):
            return self.MAX_FRAME_MEMORY
        return max(1, limit)

    def _action_context_frames(self) -> int:
        try:
            frames = max(1, int(os.getenv("LLM_ACTION_CONTEXT_FRAMES", str(self.ACTION_CONTEXT_FRAMES))))
        except ValueError:
            frames = self.ACTION_CONTEXT_FRAMES
        return min(frames, self._mm_image_limit())

    def _max_plan_actions(self) -> int:
        try:
            return max(1, min(8, int(os.getenv("LLM_MAX_PLAN_ACTIONS", str(self.MAX_PLAN_ACTIONS)))))
        except ValueError:
            return self.MAX_PLAN_ACTIONS

    def _max_significant_events(self) -> int:
        try:
            return max(
                1,
                int(os.getenv("LLM_MAX_SIGNIFICANT_EVENTS", str(self.MAX_SIGNIFICANT_EVENTS))),
            )
        except ValueError:
            return self.MAX_SIGNIFICANT_EVENTS

    def _generate_action_response(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        latest_frame: FrameData,
    ) -> tuple[str, dict[str, Any]]:
        candidate_count = self._action_candidate_count()
        if candidate_count <= 1:
            response_text = self._generate_response(
                prompt,
                frame_images,
                enable_thinking=self._action_thinking_enabled(),
            )
            parsed, response_text = self._parse_or_repair_action_json(
                prompt,
                response_text,
                frame_images,
            )
            return response_text, parsed

        responses = self._generate_responses(
            prompt,
            frame_images,
            enable_thinking=self._action_thinking_enabled(),
            choice_count=candidate_count,
        )
        candidates: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, response_text in enumerate(responses):
            try:
                parsed, full_text = self._parse_or_repair_action_json(
                    prompt,
                    response_text,
                    frame_images,
                )
                normalized = self._normalize_action_specs(parsed, latest_frame)
                if not normalized:
                    errors.append(f"candidate {index}: no usable actions")
                    continue
                candidates.append(
                    {
                        "index": index,
                        "response_text": full_text,
                        "parsed": parsed,
                        "actions": normalized,
                        "score": self._candidate_static_score(
                            parsed,
                            normalized,
                            latest_frame,
                            index,
                        ),
                    }
                )
            except Exception as exc:
                errors.append(f"candidate {index}: {type(exc).__name__}: {exc}")

        if not candidates:
            raise ValueError(
                f"All {len(responses)} model candidates failed: {errors[:4]}"
            )

        selected = max(candidates, key=lambda item: item["score"])
        if len(candidates) > 1 and self._candidate_arbiter_enabled():
            selected = self._select_candidate_with_arbiter(
                frame_images,
                latest_frame,
                candidates,
                selected,
            )
        parsed = dict(selected["parsed"])
        parsed["_candidate_selection"] = {
            "candidate_count": len(responses),
            "valid_candidates": len(candidates),
            "selected_index": selected["index"],
            "static_score": selected["score"],
            "errors": errors[:4],
        }
        return selected["response_text"], parsed

    def _parse_or_repair_action_json(
        self,
        prompt: str,
        response_text: str,
        frame_images: list[Image.Image],
    ) -> tuple[dict[str, Any], str]:
        try:
            return self._extract_action_json(response_text), response_text
        except Exception:
            repair_response = self._generate_response(
                self._build_json_repair_prompt(prompt, response_text),
                frame_images,
                enable_thinking=False,
                max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
            )
            full_text = response_text + "\n\nJSON_REPAIR_OUTPUT:\n" + repair_response
            return self._extract_action_json(repair_response), full_text

    def _action_candidate_count(self) -> int:
        try:
            count = int(os.getenv("LLM_ACTION_CANDIDATES", str(self.ACTION_CANDIDATES)))
        except ValueError:
            count = self.ACTION_CANDIDATES
        if self.game_time_remaining_s < float(os.getenv("LLM_CANDIDATE_MIN_SECONDS", "900")):
            return 1
        return max(1, min(5, count))

    def _candidate_arbiter_enabled(self) -> bool:
        value = os.getenv("LLM_CANDIDATE_ARBITER", "1").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _candidate_static_score(
        self,
        parsed: dict[str, Any],
        actions: list[dict[str, Any]],
        latest_frame: FrameData,
        index: int,
    ) -> float:
        score = 100.0 - index * 0.25
        confidence = self._coerce_confidence(parsed.get("confidence"))
        score += confidence * 8.0
        if len(actions) > 1:
            score += min(len(actions), self.MAX_PLAN_ACTIONS) * 0.35
        if actions and actions[0].get("name") == "RESET":
            score -= 25.0
        if actions and actions[0].get("name") == "ACTION6":
            score += self._click_action_score(actions[0], latest_frame)
        summary = str(parsed.get("plan_summary", "")).lower()
        assessment = str(parsed.get("board_change_assessment", "")).lower()
        useful_words = (
            "goal",
            "win",
            "complete",
            "match",
            "move",
            "collect",
            "open",
            "unlock",
            "test",
            "progress",
        )
        score += sum(0.15 for word in useful_words if word in summary or word in assessment)
        risky_words = ("random", "guess", "uncertain", "stuck", "repeat")
        score -= sum(0.5 for word in risky_words if word in summary or word in assessment)
        return score

    def _coerce_confidence(self, raw_value: Any) -> float:
        if isinstance(raw_value, (int, float)):
            return max(0.0, min(1.0, float(raw_value)))
        text = str(raw_value or "").strip().lower()
        if not text:
            return 0.5
        if text.endswith("%"):
            try:
                return max(0.0, min(1.0, float(text[:-1]) / 100.0))
            except ValueError:
                return 0.5
        labels = {
            "low": 0.25,
            "medium": 0.55,
            "med": 0.55,
            "high": 0.8,
            "certain": 0.95,
        }
        if text in labels:
            return labels[text]
        try:
            value = float(text)
            return max(0.0, min(1.0, value if value <= 1.0 else value / 100.0))
        except ValueError:
            return 0.5

    def _click_action_score(self, action_spec: dict[str, Any], latest_frame: FrameData) -> float:
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        x = self._clamp_coordinate(action_spec.get("x", 0))
        y = self._clamp_coordinate(action_spec.get("y", 0))
        if not grid or y >= len(grid) or x >= len(grid[y]):
            return -2.0
        value = int(grid[y][x])
        if value == 0:
            return -0.4
        non_zero = [
            (xx, yy)
            for yy, row in enumerate(grid)
            for xx, cell in enumerate(row)
            if int(cell) != 0
        ]
        if not non_zero:
            return 0.0
        min_x = min(xx for xx, _ in non_zero)
        max_x = max(xx for xx, _ in non_zero)
        min_y = min(yy for _, yy in non_zero)
        max_y = max(yy for _, yy in non_zero)
        margin = 2
        inside_content = min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin
        return 1.2 if inside_content else 0.2

    def _select_candidate_with_arbiter(
        self,
        frame_images: list[Image.Image],
        latest_frame: FrameData,
        candidates: list[dict[str, Any]],
        default_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        arbiter_payload = []
        for item in candidates:
            parsed = item["parsed"]
            arbiter_payload.append(
                {
                    "id": item["index"],
                    "actions": item["actions"],
                    "confidence": parsed.get("confidence"),
                    "plan_summary": str(parsed.get("plan_summary", ""))[:500],
                    "board_change_assessment": str(parsed.get("board_change_assessment", ""))[:500],
                    "static_score": round(float(item["score"]), 3),
                }
            )
        prompt = textwrap.dedent(
            f"""
            You are choosing among candidate action plans for the same ARC-AGI-3 state.
            The images are chronological with red STEP labels; the last image is current.
            Pick the plan most likely to complete the current level with few actions.
            Penalize repeated-state guesses, unsupported clicks, and reset unless necessary.

            State: {latest_frame.state.name}
            Levels completed: {latest_frame.levels_completed}
            Available actions: {self._available_model_action_names(latest_frame)}
            Ineffective in this exact state: {self._ineffective_actions_for_current_state(latest_frame)}
            Recent transitions: {json.dumps(self._prompt_history()[-4:], ensure_ascii=True)}

            Candidate plans:
            {json.dumps(arbiter_payload, ensure_ascii=True)}

            Return exactly one JSON object: {{"choice": <candidate id>, "reason": "short evidence"}}
            /no_think
            """
        ).strip()
        try:
            response_text = self._generate_response(
                prompt,
                frame_images,
                enable_thinking=False,
                max_new_tokens=self.ARBITER_MAX_NEW_TOKENS,
                json_mode=True,
            )
            parsed = self._extract_action_json(response_text)
            choice = int(parsed.get("choice"))
            for item in candidates:
                if item["index"] == choice:
                    item["parsed"] = dict(item["parsed"])
                    item["parsed"]["_arbiter"] = {
                        "choice": choice,
                        "reason": str(parsed.get("reason", ""))[:500],
                    }
                    return item
        except Exception as exc:
            logger.info("Candidate arbiter failed; using static score: %s", exc)
        return default_candidate

    def _generate_response(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        enable_thinking: bool,
        max_new_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        return self._generate_responses(
            prompt,
            frame_images,
            enable_thinking,
            max_new_tokens=max_new_tokens,
            json_mode=json_mode,
            choice_count=1,
        )[0]

    def _generate_responses(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        enable_thinking: bool,
        max_new_tokens: int | None = None,
        json_mode: bool = True,
        choice_count: int = 1,
    ) -> list[str]:
        if self._client is None or self._served_model is None:
            raise RuntimeError("vLLM client is not initialized")
        response_start = time.perf_counter()
        token_budget = max_new_tokens or int(
            os.getenv("LLM_MAX_NEW_TOKENS", str(self.MAX_NEW_TOKENS))
        )
        if max_new_tokens is None:
            # Only the main action-generation call omits max_new_tokens;
            # repair/reflection/arbiter calls pass their own explicit budget
            # and are intentionally left untouched by this.
            token_budget = self._adaptive_max_new_tokens(token_budget)
        content: list[dict[str, Any]] = []
        for frame_image in frame_images:
            image_buffer = io.BytesIO()
            frame_image.save(image_buffer, format="PNG")
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
            image_url = f"data:image/png;base64,{encoded_image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._served_model,
            "messages": messages,
            "max_tokens": token_budget,
            "temperature": float(
                os.getenv("LLM_TEMPERATURE", "0.6" if enable_thinking else "0.2")
            ),
            "top_p": float(os.getenv("LLM_TOP_P", "0.95")),
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "top_k": int(os.getenv("LLM_TOP_K", "20")),
                "repetition_penalty": float(
                    os.getenv("LLM_REPETITION_PENALTY", "1.08")
                ),
            },
        }
        if choice_count > 1:
            request_kwargs["n"] = max(1, min(5, int(choice_count)))
        # Guided JSON decoding constrains every generated token to the JSON
        # grammar from the first token. The vLLM server here is started
        # without --reasoning-parser (see _start_vllm_server), so it has no
        # way to carve out an unconstrained reasoning span before that
        # grammar kicks in -- combining the two would either silently starve
        # the model of any thinking tokens or break the completion outright.
        # _extract_action_json already scans free-form text for the JSON
        # object, so drop guided mode whenever thinking is on and lean on
        # that instead.
        if (
            json_mode
            and not enable_thinking
            and os.getenv("VLLM_JSON_MODE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            request_kwargs["response_format"] = {"type": "json_object"}
        remaining = self.game_time_remaining_s
        if remaining <= 0:
            raise TimeoutError("Per-game or global time budget exhausted before inference")
        configured_timeout = float(
            os.getenv("VLLM_REQUEST_TIMEOUT", str(self.LLM_REQUEST_TIMEOUT_S))
        )
        request_client = self._client.with_options(
            timeout=max(1.0, min(configured_timeout, remaining))
        )
        response = request_client.chat.completions.create(**request_kwargs)
        completion_tokens = (
            response.usage.completion_tokens if response.usage is not None else None
        )
        texts: list[str] = []
        finish_reasons = []
        for choice in response.choices:
            content = choice.message.content or ""
            if not isinstance(content, str):
                content = "".join(str(part) for part in content)
            texts.append(content.strip())
            finish_reasons.append(choice.finish_reason)
            if choice.finish_reason == "length":
                logger.warning(
                    "vLLM output reached token budget=%s without a stop token",
                    token_budget,
                )
        if self._timing_enabled():
            logger.info(
                "vLLM timing step=%s images=%s thinking=%s choices=%s finish=%s "
                "completion_tokens=%s budget=%s total=%.3fs",
                self.action_counter,
                [f"{image.width}x{image.height}" for image in frame_images],
                enable_thinking,
                len(texts),
                finish_reasons,
                completion_tokens,
                token_budget,
                time.perf_counter() - response_start,
            )
        return texts or [""]

    def _build_json_repair_prompt(self, original_prompt: str, bad_output: str) -> str:
        return textwrap.dedent(
            f"""
            The previous answer did not contain a valid JSON object.

            Original task:
            {original_prompt}

            Previous non-JSON answer:
            {bad_output[:3000]}

            Return exactly one JSON object now. Do not include thought, markdown, prose, or code fences.
            Required final shape:
            {{"actions": [{{"name": "up"}}, {{"name": "click", "x": 12, "y": 34}}]}}
            /no_think
            """
        ).strip()

    def _extract_action_json(self, text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        errors: list[str] = []
        command_payloads: list[tuple[int, int, dict[str, Any]]] = []
        other_payloads: list[tuple[int, int, dict[str, Any]]] = []
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, length = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if isinstance(payload, dict):
                candidate = (start, start + length, payload)
                if "actions" in payload or "action" in payload:
                    command_payloads.append(candidate)
                else:
                    other_payloads.append(candidate)
        if command_payloads:
            return max(command_payloads, key=lambda item: (item[1], -item[0]))[2]
        if other_payloads:
            return max(other_payloads, key=lambda item: (item[1], -item[0]))[2]
        raise ValueError(f"No JSON object found in model output: {text!r}; parse_errors={errors[:3]}")

    def _normalize_action_specs(
        self,
        payload: dict[str, Any],
        latest_frame: FrameData,
    ) -> list[dict[str, Any]]:
        raw_actions = payload.get("actions")
        if raw_actions is None and payload.get("action"):
            raw_actions = [payload]
        elif raw_actions is not None and not isinstance(raw_actions, list):
            raw_actions = [raw_actions]
        if not isinstance(raw_actions, list):
            return []

        normalized: list[dict[str, Any]] = []
        ineffective_actions = set(self._ineffective_actions_for_current_state(latest_frame))
        for item in raw_actions[: self._max_plan_actions()]:
            action_payload = item if isinstance(item, dict) else {}
            raw_name = self._coerce_action_name(
                action_payload.get("name") or action_payload.get("action") or item
            )
            if raw_name == "RESET":
                action = GameAction.RESET
            else:
                try:
                    action = GameAction.from_name(raw_name)
                except ValueError:
                    logger.info("Skipping unknown planned action %s", raw_name)
                    continue
            if action is not GameAction.RESET and not self._is_action_available(latest_frame, action):
                logger.info("Skipping unavailable planned action %s", raw_name)
                continue
            spec: dict[str, Any] = {"name": action.name}
            if action.is_complex():
                spec["x"] = self._clamp_coordinate(action_payload.get("x", 0))
                spec["y"] = self._clamp_coordinate(action_payload.get("y", 0))
            ineffective_key = self._action_failure_key_from_spec(spec)
            if ineffective_key in ineffective_actions:
                logger.info("Skipping action proven ineffective in current state: %s", ineffective_key)
                continue
            if action is GameAction.ACTION6 and self._click_near_failed(spec, latest_frame):
                logger.info("Skipping click near failed point in current state: %s", spec)
                continue
            normalized.append(spec)
        return normalized

    def _coerce_action_name(self, raw_name: Any) -> str:
        raw_text = str(raw_name or "").strip()
        semantic_name = raw_text.lower().replace("-", "_").replace(" ", "_")
        semantic_aliases = {
            "move_up": "up",
            "move_down": "down",
            "move_left": "left",
            "move_right": "right",
        }
        semantic_name = semantic_aliases.get(semantic_name, semantic_name)
        if semantic_name in self.MODEL_TO_GAME_ACTION:
            return self.MODEL_TO_GAME_ACTION[semantic_name]

        text = raw_text.upper()
        if not text:
            return ""
        if text.isdigit():
            try:
                return GameAction.from_id(int(text)).name
            except ValueError:
                return ""
        digit_match = re.fullmatch(r"ACTION[_\s-]*(\d+)", text)
        if digit_match:
            return f"ACTION{digit_match.group(1)}"
        return text

    def _dequeue_action(
        self,
        latest_frame: FrameData,
        extra_reasoning: dict[str, Any] | None = None,
        remember: bool = True,
    ) -> GameAction:
        spec = self.pending_actions.pop(0)
        action = self._materialize_action(spec, latest_frame)
        reasoning: dict[str, Any] = {
            "driver": "vllm-openai-compatible",
            "model": self._served_model,
            "thinking_enabled": self._action_thinking_enabled(),
            "from_plan_queue": True,
            "remaining_planned_actions": len(self.pending_actions),
            "plan_summary": self.last_plan_summary,
            "raw_plan_action": spec,
            "available_actions": list(latest_frame.available_actions or []),
        }
        if extra_reasoning:
            reasoning.update(extra_reasoning)
        action.reasoning = reasoning
        if remember:
            self._remember_step(latest_frame, action, "DEQUEUED_FROM_PLAN", self._action_to_payload(action))
        logger.info(
            "Dequeued planned action %s for %s (%s remaining)",
            action.name,
            self.game_id,
            len(self.pending_actions),
        )
        return action

    def _materialize_action(self, spec: dict[str, Any], latest_frame: FrameData) -> GameAction:
        raw_name = str(spec.get("name", "")).upper().strip()
        action = GameAction.RESET if raw_name == "RESET" else GameAction.from_name(raw_name)
        if action is not GameAction.RESET and not self._is_action_available(latest_frame, action):
            self.pending_actions = []
            return self._fallback_action(latest_frame, f"Planned action {action.name} no longer available.")
        if self._model_action_name(action) in self._ineffective_actions_for_current_state(latest_frame):
            self.pending_actions = []
            return self._fallback_action(
                latest_frame, f"Planned action {action.name} already failed in this state."
            )
        if action.is_complex():
            if self._click_near_failed(spec, latest_frame):
                self.pending_actions = []
                return self._fallback_action(
                    latest_frame,
                    f"Planned click near failed point in this state: {spec}.",
                )
            action.set_data(
                {
                    "x": self._clamp_coordinate(spec.get("x", 0)),
                    "y": self._clamp_coordinate(spec.get("y", 0)),
                }
            )
        return action

    def _action_to_payload(self, action: GameAction) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action.name}
        payload.update(self._action_data_dict(action))
        return payload

    def _action_data_dict(self, action: GameAction | None) -> dict[str, Any]:
        if action is None:
            return {}
        action_data = getattr(action, "action_data", None)
        if hasattr(action_data, "model_dump"):
            raw_data = action_data.model_dump()
        elif isinstance(action_data, dict):
            raw_data = action_data
        else:
            raw_data = {}
        return {
            key: self._clamp_coordinate(raw_data[key])
            for key in ("x", "y")
            if key in raw_data
        }

    def _action_failure_key_from_spec(self, spec: dict[str, Any]) -> str:
        action_name = str(spec.get("name", "")).upper().strip()
        if action_name == "ACTION6":
            x = self._clamp_coordinate(spec.get("x", 0))
            y = self._clamp_coordinate(spec.get("y", 0))
            return f"click@{x},{y}"
        try:
            return self._model_action_name(GameAction.from_name(action_name))
        except ValueError:
            return action_name.lower()

    def _action_failure_key(self, action: GameAction) -> str:
        if action.is_complex():
            data = self._action_data_dict(action)
            return f"click@{data.get('x', 0)},{data.get('y', 0)}"
        return self._model_action_name(action)

    def _abstraction_enabled(self) -> bool:
        return self._env_flag("LLM_STATE_ABSTRACTION", "1")

    def _abstract_failure_threshold(self) -> int:
        try:
            return max(1, int(os.getenv("LLM_ABSTRACT_FAIL_THRESHOLD", "2")))
        except ValueError:
            return 2

    def _ineffective_actions_for_current_state(
        self, latest_frame: FrameData
    ) -> list[str]:
        failed = getattr(self, "failed_state_actions", {})
        keys = set(failed.get(self._frame_hash(latest_frame.frame), set()))
        if self._abstraction_enabled():
            # An action only counts as ineffective under the coarse key once it
            # has failed there repeatedly, so a single coincidence cannot ban a
            # legitimate action for the rest of the level.
            threshold = self._abstract_failure_threshold()
            abstract = getattr(self, "failed_abstract_actions", {})
            counts = abstract.get(self._state_abstraction(latest_frame.frame), {})
            keys.update(key for key, count in counts.items() if count >= threshold)
        return sorted(keys)

    def _build_reflection_prompt(self, latest_frame: FrameData) -> str:
        reflection_interval = self._reflection_interval()
        transitions = json.dumps(
            self.reflection_buffer[-reflection_interval :], ensure_ascii=True
        )
        # significant_events survives the whole game (reflection_buffer gets
        # drained every reflection_interval steps), but is only ever the rare
        # "level advanced" / "genuinely new state" steps, so a bounded slice
        # here stays cheap even though the underlying list can hold up to
        # MAX_SIGNIFICANT_EVENTS entries.
        significant_history = json.dumps(
            self.significant_events[-20:], ensure_ascii=True
        )
        significant_block = ""
        if self.significant_events:
            significant_block = (
                "\n\nEarlier significant events this game (level-ups and "
                f"genuinely new states, may be from prior levels):\n{significant_history}"
            )
        shared_mechanism_block = ""
        if self.shared_mechanisms:
            shared_mechanism_block = (
                "\n\nCross-game notes (unverified in this game, treat as hints,"
                f" not facts):\n{self.shared_mechanisms[: self.MAX_SHARED_MECHANISM_PROMPT_CHARS]}"
            )
        return textwrap.dedent(
            f"""
            You are the reflection agent for an ARC-AGI-3 game. Review the previous
            memory, the last {reflection_interval} completed transitions, and the
            chronological images. The final image is current. Pixel changes may be movement, transformation, collection,
            animation, or UI, so do not assume translation.

            Keep only evidence-supported, useful conclusions. Correct stale beliefs.
            Under ## Rules, tag each item [CONFIRMED] (supported by multiple
            transitions), [HYPOTHESIS] (a single observation or inference), or
            [FALSIFIED] (contradicted by recent transitions, no longer trusted).
            Before keeping a [HYPOTHESIS] from previous memory, check whether the
            last transitions confirm or contradict it; if contradicted, mark it
            [FALSIFIED] and say what changed. State a concrete next goal.
            Return only a compact Markdown document under {self.MAX_REFLECTION_CHARS}
            characters with exactly these headings:

            # Agent Memory
            ## Rules
            ## Goal
            ## Progress
            ## Avoid

            Previous memory:
            {self.reflection_memory}{shared_mechanism_block}

            Current level: {int(latest_frame.levels_completed) + 1}
            Completed transitions:
            {transitions}{significant_block}
            /no_think
            """
        ).strip()

    def _clean_reflection_markdown(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return self.reflection_memory
        if not cleaned.startswith("# Agent Memory"):
            cleaned = "# Agent Memory\n\n" + cleaned
        return cleaned[: self.MAX_REFLECTION_CHARS].rstrip()

    def _save_reflection_memory(self) -> None:
        try:
            memory_dir = os.path.dirname(self.reflection_memory_path)
            if memory_dir:
                os.makedirs(memory_dir, exist_ok=True)
            temp_path = self.reflection_memory_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as memory_file:
                memory_file.write(self.reflection_memory + "\n")
            os.replace(temp_path, self.reflection_memory_path)
        except OSError as exc:
            logger.warning("Failed to save reflection memory: %s", exc)

    def _run_reflection(self, latest_frame: FrameData) -> None:
        reflection_interval = self._reflection_interval()
        if not reflection_interval or len(self.reflection_buffer) < reflection_interval:
            return
        # A reflection may revise the goal, so discard any stale queued plan.
        self.pending_actions = []
        # Pick up anything another concurrently-running game confirmed since
        # this instance last loaded it (Swarm runs one thread per game
        # against the same process), before this game's own prompt is built.
        self.shared_mechanisms = self._load_shared_mechanisms()
        prompt = self._build_reflection_prompt(latest_frame)
        # Never exceed the server's --limit-mm-per-prompt image cap; the older
        # transitions are already summarised as text in the reflection prompt.
        images = self._build_context_images(
            latest_frame.frame,
            limit=min(reflection_interval + 1, self._mm_image_limit()),
        )
        try:
            response = self._generate_response(
                prompt,
                images,
                enable_thinking=False,
                max_new_tokens=self._reflection_max_new_tokens(),
                json_mode=False,
            )
            self.reflection_memory = self._clean_reflection_markdown(response)
            self._save_reflection_memory()
            self.reflections_completed += 1
            if not any(
                tag in self.reflection_memory
                for tag in ("[CONFIRMED]", "[HYPOTHESIS]", "[FALSIFIED]")
            ):
                # Observability only -- the tagging instruction in
                # _build_reflection_prompt is a nudge, not a hard-gate (Tycho's
                # own data shows forcing a rigid trigger rule underperforms
                # leaving the model discretion), so a missing tag never blocks
                # or rewrites the reflection, it's just worth knowing about.
                logger.info(
                    "Reflection for %s produced no [CONFIRMED]/[HYPOTHESIS]/"
                    "[FALSIFIED] tags",
                    self.game_id,
                )
            self._update_shared_mechanisms()
            logger.info(
                "Reflection completed for %s after %s transitions; memory=%s",
                self.game_id,
                reflection_interval,
                self.reflection_memory_path,
            )
        except Exception as exc:
            # A silent reflection failure leaves reflection_memory at its
            # placeholder for the whole run, so make the count observable.
            self.reflection_failures += 1
            logger.warning(
                "Reflection failed for %s (%s consecutive-run failures): %s",
                self.game_id,
                self.reflection_failures,
                exc,
            )
        finally:
            del self.reflection_buffer[:reflection_interval]

    def _remember_step(
        self,
        latest_frame: FrameData,
        action: GameAction,
        raw_text: str,
        parsed: dict[str, Any],
    ) -> None:
        item = {
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "available_actions": self._available_model_action_names(latest_frame),
            "chosen_action": self._model_action_name(action),
            "action_data": self._action_data_dict(action) or None,
            "failure_key": self._action_failure_key(action),
            "raw_model_output": raw_text[:400],
            "parsed_output": parsed,
            "reasoning": parsed.get("reasoning", ""),
            "plan_before_action": parsed.get("plan_summary", ""),
            "frame_signature": self._frame_signature(latest_frame.frame),
        }
        self.history.append(item)
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY :]

    def _write_llm_trace(
        self,
        latest_frame: FrameData,
        prompt: str,
        response_text: str,
        parsed: dict[str, Any] | None = None,
        chosen_action: GameAction | None = None,
        context_images: list[Image.Image] | None = None,
        error: str | None = None,
    ) -> None:
        trace_path = os.getenv("LLM_TRACE_PATH", self.DEFAULT_TRACE_PATH)
        action_data = (self._action_data_dict(chosen_action) or None) if chosen_action else None
        context_image_paths = self._save_trace_images(trace_path, context_images)
        record = {
            "timestamp": time.time(),
            "game_id": self.game_id,
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "available_actions": self._available_action_names(latest_frame),
            "frame_signature": self._frame_signature(latest_frame.frame),
            "reflection_memory_path": self.reflection_memory_path,
            "reflection_memory": self.reflection_memory,
            "reflections_completed": self.reflections_completed,
            "reflection_failures": self.reflection_failures,
            "state_abstraction": self._state_abstraction(latest_frame.frame),
            "input": {
                "prompt": prompt,
                "images": {
                    "source": "separate chronological observation frames, oldest first",
                    "paths": context_image_paths,
                    "format": "ordered PIL RGB images with red STEP labels",
                    "scale": self.FRAME_IMAGE_SCALE,
                },
            },
            "output": {
                "raw_text": response_text,
                "parsed_json": parsed,
                "plan_summary": parsed.get("plan_summary", "") if parsed else "",
                "planned_actions": parsed.get("actions") if parsed else None,
                "chosen_action": (
                    self._model_action_name(chosen_action) if chosen_action else None
                ),
                "game_action": chosen_action.name if chosen_action else None,
                "action_data": action_data,
                "pending_actions_after_choice": self.pending_actions,
            },
            "error": error,
        }
        try:
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write LLM trace JSON: %s", exc)

    def _timing_enabled(self) -> bool:
        value = os.getenv("LLM_TIMING", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _action_thinking_enabled(self) -> bool:
        value = os.getenv("LLM_ACTION_THINKING", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _log_timing(
        self,
        latest_frame: FrameData,
        context_images: list[Image.Image],
        timings: dict[str, float],
    ) -> None:
        if not self._timing_enabled():
            return
        ordered = [
            "observe_frame",
            "build_prompt",
            "build_context_images",
            "generate_response",
            "extract_json",
            "repair_generate_response",
            "repair_extract_json",
            "plan_to_action",
            "write_trace",
            "remember_step",
            "total_choose_action",
        ]
        timing_text = " ".join(
            f"{name}={timings[name]:.3f}s" for name in ordered if name in timings
        )
        logger.info(
            "Agent timing step=%s state=%s levels=%s images=%s %s",
            self.action_counter,
            latest_frame.state.name,
            latest_frame.levels_completed,
            [f"{image.width}x{image.height}" for image in context_images],
            timing_text,
        )

    def _observe_frame(self, latest_frame: FrameData) -> None:
        level_number = int(latest_frame.levels_completed) + 1
        if level_number != self.current_level_number:
            self.current_level_number = level_number
            self.failed_state_actions = {}
            self.failed_abstract_actions = {}
            self.pending_actions = []
        current_hash = self._frame_hash(latest_frame.frame)
        current_entry = {
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "frame": latest_frame.frame,
            "frame_hash": current_hash,
            "state_abstraction": self._state_abstraction(latest_frame.frame),
            "frame_signature": self._frame_signature(latest_frame.frame),
        }

        if self.frame_memory and self.frame_memory[-1]["step"] == self.action_counter:
            self.frame_memory[-1] = current_entry
        else:
            self.frame_memory.append(current_entry)
            if len(self.frame_memory) > self.MAX_FRAME_MEMORY:
                self.frame_memory = self.frame_memory[-self.MAX_FRAME_MEMORY :]

        if not self.history:
            return
        previous_action = self.history[-1]
        if "after_frame_signature" in previous_action:
            return
        if previous_action["step"] >= self.action_counter:
            return

        before_entry = None
        for item in reversed(self.frame_memory[:-1]):
            if item["step"] == previous_action["step"]:
                before_entry = item
                break
        if before_entry is None and len(self.frame_memory) >= 2:
            before_entry = self.frame_memory[-2]
        if before_entry is None:
            return

        changed_pixels = self._changed_pixels(before_entry["frame"], latest_frame.frame)
        levels_delta = latest_frame.levels_completed - previous_action["levels_completed"]
        repeated_state = any(
            item["frame_hash"] == current_hash for item in self.frame_memory[:-1]
        )
        if changed_pixels == 0 and levels_delta == 0:
            failed_actions = getattr(self, "failed_state_actions", None)
            if failed_actions is None:
                self.failed_state_actions = {}
                failed_actions = self.failed_state_actions
            failure_key = (
                previous_action.get("failure_key") or previous_action["chosen_action"]
            )
            failed_actions.setdefault(before_entry["frame_hash"], set()).add(failure_key)
            abstract_key = before_entry.get("state_abstraction")
            if abstract_key:
                if getattr(self, "failed_abstract_actions", None) is None:
                    self.failed_abstract_actions = {}
                counts = self.failed_abstract_actions.setdefault(abstract_key, {})
                counts[failure_key] = counts.get(failure_key, 0) + 1
            self.pending_actions = []
        elif repeated_state and levels_delta == 0:
            self.pending_actions = []
        previous_action.update(
            {
                "after_step": self.action_counter,
                "after_state": latest_frame.state.name,
                "after_levels_completed": latest_frame.levels_completed,
                "after_frame_signature": self._frame_signature(latest_frame.frame),
                "after_frame_hash": current_hash,
                "changed_pixels": changed_pixels,
                "levels_delta": levels_delta,
                "state_changed": before_entry["frame_hash"] != current_hash,
                "repeated_state": repeated_state,
            }
        )
        self.reflection_buffer.append(self._compact_history_item(previous_action))
        if levels_delta != 0 or (previous_action["state_changed"] and not repeated_state):
            self.significant_events.append(self._compact_history_item(previous_action))
            max_events = self._max_significant_events()
            if len(self.significant_events) > max_events:
                self.significant_events = self.significant_events[-max_events:]

    def _compact_history_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": item.get("step"),
            "action": item.get("chosen_action"),
            "action_data": item.get("action_data"),
            "failure_key": item.get("failure_key"),
            "levels_before": item.get("levels_completed"),
            "levels_after": item.get("after_levels_completed"),
            "levels_delta": item.get("levels_delta"),
            "changed_pixels": item.get("changed_pixels"),
            "state_changed": item.get("state_changed"),
            "repeated_state": item.get("repeated_state"),
            "plan_before_action": item.get("plan_before_action", ""),
            "frame_before": item.get("frame_signature"),
            "frame_after": item.get("after_frame_signature"),
        }

    def _prompt_history(self) -> list[dict[str, Any]]:
        return [
            self._compact_history_item(item)
            for item in self.history[-self.MAX_HISTORY :]
        ]

    def _save_trace_images(
        self,
        trace_path: str,
        context_images: list[Image.Image] | None,
    ) -> list[str]:
        if not context_images:
            return []
        trace_images = os.getenv("LLM_TRACE_IMAGES", "0").strip().lower()
        if trace_images not in {"1", "true", "yes", "on"}:
            return []
        try:
            base_dir = os.getenv("LLM_TRACE_IMAGE_DIR")
            if not base_dir:
                trace_dir = os.path.dirname(trace_path) or "."
                base_dir = os.path.join(trace_dir, "llm_trace_images")
            os.makedirs(base_dir, exist_ok=True)
            safe_game_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.game_id)
            image_paths = []
            for index, context_image in enumerate(context_images):
                image_path = os.path.join(
                    base_dir,
                    f"{safe_game_id}_step_{self.action_counter:04d}_frame_{index:02d}.png",
                )
                context_image.save(image_path)
                image_paths.append(image_path)
            return image_paths
        except Exception as exc:
            logger.warning("Failed to save LLM trace images: %s", exc)
            return []

    def _fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        ineffective_actions = set(self._ineffective_actions_for_current_state(latest_frame))
        available = [
            action
            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
                GameAction.ACTION6,
                GameAction.ACTION7,
            ]
            if self._is_action_available(latest_frame, action)
            and self._model_action_name(action) not in ineffective_actions
        ]
        if not available:
            available = [
                action
                for action in [
                    GameAction.ACTION1,
                    GameAction.ACTION2,
                    GameAction.ACTION3,
                    GameAction.ACTION4,
                    GameAction.ACTION5,
                    GameAction.ACTION6,
                    GameAction.ACTION7,
                ]
                if self._is_action_available(latest_frame, action)
            ]
        if not available:
            action = GameAction.ACTION5
            action.reasoning = {"fallback": True, "reason": reason, "note": "No availability metadata"}
            return action

        action = available[self.action_counter % len(available)]
        if action.is_complex():
            x, y = self._pick_interesting_coordinate(latest_frame.frame, latest_frame)
            action.set_data({"x": x, "y": y})
        action.reasoning = {
            "fallback": True,
            "reason": reason,
            "strategy": "ordered_legal_action_cycle",
        }
        return action

    def _pick_interesting_coordinate(
        self,
        frame_3d: list[list[list[Any]]],
        latest_frame: FrameData | None = None,
    ) -> tuple[int, int]:
        last_grid = frame_3d[-1] if frame_3d else []
        non_zero = []
        failed_points = (
            self._failed_click_points_for_current_state(latest_frame)
            if latest_frame is not None
            else []
        )
        radius = self._click_failure_radius()
        for y, row in enumerate(last_grid[:64]):
            for x, value in enumerate(row[:64]):
                if int(value) != 0:
                    non_zero.append((x, y))
        if failed_points and radius > 0:
            filtered = [
                (x, y)
                for x, y in non_zero
                if all(abs(x - fx) > radius or abs(y - fy) > radius for fx, fy in failed_points)
            ]
            if filtered:
                non_zero = filtered
        if non_zero:
            component_centers = self._component_centers(set(non_zero))
            if component_centers:
                index = self.action_counter % len(component_centers)
                return component_centers[index]
            return non_zero[self.action_counter % len(non_zero)]
        return self.rng.randint(0, 63), self.rng.randint(0, 63)

    def _component_centers(
        self,
        allowed_points: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        components: list[tuple[int, int, int, int, int, int]] = []
        for start in sorted(allowed_points):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            cells: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    point = (nx, ny)
                    if point in seen or point not in allowed_points:
                        continue
                    seen.add(point)
                    stack.append(point)
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            area = len(cells)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            center_x = round(sum(xs) / area)
            center_y = round(sum(ys) / area)
            components.append((area, (max_x - min_x + 1) * (max_y - min_y + 1), center_x, center_y, min_x, min_y))
        components.sort()
        return [(center_x, center_y) for _area, _bbox_area, center_x, center_y, _min_x, _min_y in components]

    def _available_action_names(self, latest_frame: FrameData) -> list[str]:
        available_actions = latest_frame.available_actions or []
        if not available_actions:
            return [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
                "ACTION7",
            ]
        names = []
        for item in available_actions:
            value = int(item.value) if hasattr(item, "value") else int(item)
            names.append(f"ACTION{value}")
        return names

    def _available_model_action_names(self, latest_frame: FrameData) -> list[str]:
        return [
            self.GAME_TO_MODEL_ACTION.get(name, name.lower())
            for name in self._available_action_names(latest_frame)
        ]

    def _model_action_name(self, action: GameAction) -> str:
        return self.GAME_TO_MODEL_ACTION.get(action.name, action.name.lower())

    def _legal_action_instructions(self, available_actions: list[str]) -> str:
        descriptions = {
            "up": '- {"name":"up"}: move up',
            "down": '- {"name":"down"}: move down',
            "left": '- {"name":"left"}: move left',
            "right": '- {"name":"right"}: move right',
            "spacebar": '- {"name":"spacebar"}: activate/confirm',
            "click": '- {"name":"click","x":12,"y":34}: click original grid coordinates, x/y integers in [0,63], not scaled image pixels or STEP-label pixels',
            "undo": '- {"name":"undo"}: undo/reverse',
        }
        return "\n".join(descriptions[action] for action in available_actions if action in descriptions)

    def _frame_descriptor(self, frame_3d: list[list[list[Any]]]) -> dict[str, Any]:
        grid = frame_3d[-1] if frame_3d else []
        if not grid:
            return {"height": 0, "width": 0, "colors": {}}
        height = len(grid)
        width = max((len(row) for row in grid), default=0)
        components_by_colour = self._connected_components(grid)
        colors: dict[str, dict[str, Any]] = {}
        for colour, colour_components in components_by_colour.items():
            if not colour_components:
                continue
            bbox = [
                min(comp["min_x"] for comp in colour_components),
                min(comp["min_y"] for comp in colour_components),
                max(comp["max_x"] for comp in colour_components),
                max(comp["max_y"] for comp in colour_components),
            ]
            areas = sorted((comp["area"] for comp in colour_components), reverse=True)
            colors[str(colour)] = {
                "count": sum(areas),
                "bbox": bbox,
                "sample": [
                    [comp["centroid_x"], comp["centroid_y"]] for comp in colour_components[:6]
                ],
                "components": len(colour_components),
                "component_areas": areas[:6],
            }
        top_colors = sorted(colors.items(), key=lambda pair: pair[1]["count"], reverse=True)[:10]
        descriptor: dict[str, Any] = {
            "height": height,
            "width": width,
            "nonzero_colors": {key: value for key, value in top_colors},
        }
        adjacency = self._component_adjacency(components_by_colour)
        if adjacency:
            descriptor["adjacent_component_pairs"] = adjacency[:12]
        return descriptor

    def _is_action_available(self, latest_frame: FrameData, action: GameAction) -> bool:
        available_actions = latest_frame.available_actions or []
        if not available_actions:
            return action is not GameAction.RESET
        available_ids = {
            int(item.value) if hasattr(item, "value") else int(item)
            for item in available_actions
        }
        return int(action.value) in available_ids

    def _clamp_coordinate(self, value: Any) -> int:
        try:
            coord = int(value)
        except (TypeError, ValueError):
            coord = 0
        return max(0, min(63, coord))

    def _click_failure_radius(self) -> int:
        try:
            return max(0, int(os.getenv("LLM_CLICK_FAILURE_RADIUS", "0")))
        except ValueError:
            return 0

    def _failed_click_points_for_current_state(
        self, latest_frame: FrameData | None
    ) -> list[tuple[int, int]]:
        if latest_frame is None:
            return []
        points: list[tuple[int, int]] = []
        for key in self._ineffective_actions_for_current_state(latest_frame):
            match = re.fullmatch(r"click@(\d+),(\d+)", key)
            if match:
                points.append((int(match.group(1)), int(match.group(2))))
        return points

    def _click_near_failed(
        self,
        spec: dict[str, Any],
        latest_frame: FrameData,
    ) -> bool:
        radius = self._click_failure_radius()
        if radius <= 0:
            return False
        x = self._clamp_coordinate(spec.get("x", 0))
        y = self._clamp_coordinate(spec.get("y", 0))
        for failed_x, failed_y in self._failed_click_points_for_current_state(latest_frame):
            if abs(x - failed_x) <= radius and abs(y - failed_y) <= radius:
                return True
        return False

    def _frame_signature(self, frame_3d: list[list[list[Any]]]) -> dict[str, Any]:
        last_grid = frame_3d[-1] if frame_3d else []
        if not last_grid:
            return {"height": 0, "width": 0, "non_zero": 0}
        height = len(last_grid)
        width = len(last_grid[0]) if last_grid[0] else 0
        non_zero = sum(1 for row in last_grid for value in row if int(value) != 0)
        return {"height": height, "width": width, "non_zero": non_zero}

    def _frame_hash(self, frame_3d: list[list[list[Any]]]) -> str:
        grid = frame_3d[-1] if frame_3d else []
        payload = json.dumps(
            self._comparison_grid(grid), separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _connected_components(
        grid: list[list[Any]],
    ) -> dict[int, list[dict[str, int]]]:
        """4-connected flood-fill components, grouped by non-zero colour.

        Shared by `_state_abstraction` (its stable per-colour layout key) and
        `_frame_descriptor` (the prompt-facing segmentation summary) so the
        two never drift into two different definitions of "component".
        """
        by_colour: dict[int, set[tuple[int, int]]] = {}
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                try:
                    colour = int(value)
                except (TypeError, ValueError):
                    continue
                if colour != 0:
                    by_colour.setdefault(colour, set()).add((x, y))

        components: dict[int, list[dict[str, int]]] = {}
        for colour, points in by_colour.items():
            remaining = set(points)
            colour_components: list[dict[str, int]] = []
            while remaining:
                stack = [remaining.pop()]
                cells: list[tuple[int, int]] = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if neighbour in remaining:
                            remaining.discard(neighbour)
                            stack.append(neighbour)
                area = len(cells)
                xs = [x for x, _ in cells]
                ys = [y for _, y in cells]
                colour_components.append(
                    {
                        "area": area,
                        "min_x": min(xs),
                        "min_y": min(ys),
                        "max_x": max(xs),
                        "max_y": max(ys),
                        "centroid_x": sum(xs) // area,
                        "centroid_y": sum(ys) // area,
                    }
                )
            components[colour] = colour_components
        return components

    @staticmethod
    def _component_adjacency(
        components_by_colour: dict[int, list[dict[str, int]]],
        margin: int = 2,
        max_components: int = 150,
    ) -> list[dict[str, int]]:
        """Component pairs whose bounding boxes come within `margin` pixels.

        A cheap bbox-gap proxy for "these two shapes are near each other" --
        not a real pixel distance, but good enough to hint the model at
        possible contact/interaction without a full geometry computation.
        """
        flat: list[tuple[int, dict[str, int]]] = [
            (colour, comp)
            for colour, comps in components_by_colour.items()
            for comp in comps
        ]
        if len(flat) > max_components:
            # All-pairs below is O(n^2); a noisy/checkerboard-textured frame
            # can produce thousands of single-pixel "components" (measured:
            # ~5.3s for a full 64x64 checkerboard, unbounded). Keep only the
            # largest -- most likely to be actual game objects rather than
            # noise -- so a single step can never blow the time budget.
            flat.sort(key=lambda item: item[1]["area"], reverse=True)
            flat = flat[:max_components]
        pairs: list[dict[str, int]] = []
        for i in range(len(flat)):
            colour_a, comp_a = flat[i]
            for j in range(i + 1, len(flat)):
                colour_b, comp_b = flat[j]
                gap_x = max(
                    comp_a["min_x"] - comp_b["max_x"],
                    comp_b["min_x"] - comp_a["max_x"],
                    0,
                )
                gap_y = max(
                    comp_a["min_y"] - comp_b["max_y"],
                    comp_b["min_y"] - comp_a["max_y"],
                    0,
                )
                if gap_x <= margin and gap_y <= margin:
                    pairs.append(
                        {"color_a": colour_a, "color_b": colour_b, "gap": max(gap_x, gap_y)}
                    )
        pairs.sort(key=lambda pair: pair["gap"])
        return pairs

    def _state_abstraction(self, frame_3d: list[list[list[Any]]]) -> str:
        """Coarse layout key that survives animations and step counters.

        Per-colour connected components, keyed by colour, area bucket and a
        quantised centroid. A blinking pixel or a ticking digit perturbs the
        exact frame hash on every step but leaves this key stable, which is
        what lets failed-action memory accumulate across a level.
        """
        grid = self._comparison_grid(frame_3d[-1] if frame_3d else [])
        components_by_colour = self._connected_components(grid)

        descriptors: list[tuple[int, int, int, int]] = []
        for colour, colour_components in components_by_colour.items():
            for comp in colour_components:
                area = comp["area"]
                if area < 2:
                    # Single stray pixels are usually cursors or UI noise.
                    continue
                descriptors.append(
                    (
                        colour,
                        self._area_bucket(area),
                        comp["centroid_x"] // 2,
                        comp["centroid_y"] // 2,
                    )
                )

        payload = json.dumps(sorted(descriptors), separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _area_bucket(area: int) -> int:
        return area.bit_length()

    def _border_ignore_pixels(self) -> int:
        raw_value = os.getenv("LLM_FRAME_BORDER_IGNORE", str(self.FRAME_BORDER_IGNORE))
        try:
            return max(0, int(raw_value))
        except ValueError:
            return self.FRAME_BORDER_IGNORE

    def _comparison_grid(self, grid: list[list[Any]]) -> list[list[Any]]:
        border = self._border_ignore_pixels()
        if border == 0 or len(grid) <= border * 2:
            return grid
        trimmed = []
        for row in grid[border:-border]:
            if len(row) <= border * 2:
                trimmed.append(row)
            else:
                trimmed.append(row[border:-border])
        return trimmed

    def _changed_pixels(
        self,
        before_3d: list[list[list[Any]]],
        after_3d: list[list[list[Any]]],
    ) -> int:
        before = self._comparison_grid(before_3d[-1] if before_3d else [])
        after = self._comparison_grid(after_3d[-1] if after_3d else [])
        height = max(len(before), len(after))
        width = max(
            max((len(row) for row in before), default=0),
            max((len(row) for row in after), default=0),
        )
        changed = 0
        for y in range(height):
            before_row = before[y] if y < len(before) else []
            after_row = after[y] if y < len(after) else []
            for x in range(width):
                before_value = before_row[x] if x < len(before_row) else 0
                after_value = after_row[x] if x < len(after_row) else 0
                if before_value != after_value:
                    changed += 1
        return changed

    def _frame_to_image(self, frame_3d: list[list[list[Any]]]) -> Image.Image:
        last_grid = frame_3d[-1] if frame_3d else []
        if not last_grid:
            last_grid = [[0 for _ in range(64)] for _ in range(64)]

        height = max(len(last_grid), 1)
        width = max(max((len(row) for row in last_grid), default=0), 1)
        image = Image.new("RGB", (width, height), self.ARC_PALETTE[0])
        pixels = []
        for y in range(height):
            row = last_grid[y] if y < len(last_grid) else []
            for x in range(width):
                value = row[x] if x < len(row) else 0
                try:
                    color_index = int(value) % len(self.ARC_PALETTE)
                except (TypeError, ValueError):
                    color_index = 0
                pixels.append(self.ARC_PALETTE[color_index])
        image.putdata(pixels)

        if self.FRAME_IMAGE_SCALE > 1:
            resampling = getattr(Image, "Resampling", Image).NEAREST
            image = image.resize(
                (image.width * self.FRAME_IMAGE_SCALE, image.height * self.FRAME_IMAGE_SCALE),
                resampling,
            )
        return image

    def _build_context_images(
        self,
        latest_frame_3d: list[list[list[Any]]],
        limit: int | None = None,
    ) -> list[Image.Image]:
        frame_limit = limit or self._action_context_frames()
        recent_entries = self.frame_memory[-frame_limit:] or [
            {"step": self.action_counter, "frame": latest_frame_3d}
        ]
        return [
            self._label_image(self._frame_to_image(item["frame"]), f"STEP {item['step']}")
            for item in recent_entries
        ]

    def _label_font(self, image_font: Any) -> Any:
        try:
            return image_font.truetype("DejaVuSans-Bold.ttf", 32)
        except OSError:
            try:
                return image_font.load_default(size=32)
            except TypeError:
                return image_font.load_default()

    def _label_image(self, image: Image.Image, label: str) -> Image.Image:
        labeled = image.copy()
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(labeled)
            draw.text(
                (8, 6),
                label,
                font=self._label_font(ImageFont),
                fill=(255, 32, 32),
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
            return labeled
        except Exception as exc:
            logger.warning("Falling back to bitmap STEP label: %s", exc)
            self._draw_bitmap_text(labeled, label.upper(), 4, 4, 2)
            return labeled

    def _draw_bitmap_text(
        self,
        image: Image.Image,
        text: str,
        x: int,
        y: int,
        scale: int,
        color: tuple[int, int, int] = (255, 65, 54),
    ) -> None:
        cursor = x
        for char in text:
            glyph = self.LABEL_GLYPHS.get(char, self.LABEL_GLYPHS[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit != "1":
                        continue
                    left = cursor + gx * scale
                    top = y + gy * scale
                    for py in range(scale):
                        for px in range(scale):
                            xx = left + px
                            yy = top + py
                            if 0 <= xx < image.width and 0 <= yy < image.height:
                                image.putpixel((xx, yy), color)
            cursor += 4 * scale
            if cursor >= image.width:
                break

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
        return "\n".join(lines)
