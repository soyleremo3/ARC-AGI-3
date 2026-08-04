"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

Mirrors the structure of the actual working Kaggle submission (the
vLLM-driven `forge_v46` harness), not just the trivial starter loop:

  Cell 0: markdown banner.
  Cell 1: unpack the offline vLLM wheelhouse and install it, driven by
          PROFILE_ENV below.
  Cell 2: install the `arc-agi` wheel from the offline competition dataset.
  Cell 3: write `my_agent.py` to /tmp/ — its body is THIS file
          (agent/my_agent.py).
  Cell 4: commit-mode guardrail — smoke-import MyAgent under the offline
          framework and, unless skipped, actually boot vLLM once and run
          one generation request before risking a competition rerun.
  Cell 5: if a gateway sidecar is reachable (or this is the competition
          rerun), copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`. If
          RUN_ARC_LOCAL_VALIDATION is set, run the same thing against a
          local offline Arcade server instead. Otherwise skip.
  Cell 6: during commit / save-and-run-all, write a dummy submission.parquet
          so Kaggle accepts the commit even though no gateway was reachable.

You don't normally need to call this directly — `make submit` runs it for
you. PROFILE_ENV is duplicated verbatim into cells 1, 4 and 5 (each cell is
independently self-contained, matching how Kaggle notebooks are actually
executed cell-by-cell) — edit it in ONE place here and every cell picks up
the change on the next `make notebook` / `make submit`.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for a non-ML agent.
#   "t4"       — Nvidia T4 ×2. Small models, fast iteration.
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX PRO 6000. ARC-AGI-3 exclusive, burns GPU quota
#                faster. Required for this agent: it serves a 31B-parameter
#                model (gemma-4-31b-it) through vLLM, which does not fit on
#                t4/p100.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# Model / agent behaviour knobs, read as env vars by agent/my_agent.py at
# runtime. Change values here, not inside the generated notebook cells.
PROFILE_ENV = {
    "ARC_AGENT_NAME": "forge_v46_gemma31b_public_single",
    "ARC_MODEL_PROFILE": "gemma31b_public_single",
    "AGENT_MAX_ACTIONS": "1000",
    "LLM_ACTION_CANDIDATES": "1",
    "LLM_ACTION_CONTEXT_FRAMES": "4",
    "LLM_ACTION_THINKING": "1",
    "LLM_CANDIDATE_ARBITER": "0",
    "LLM_CLICK_FAILURE_RADIUS": "0",
    "LLM_CONFIDENCE_PROMPT": "0",
    "LLM_INCLUDE_FRAME_DESCRIPTOR": "0",
    "LLM_MAX_NEW_TOKENS": "3072",
    "LLM_MAX_PLAN_ACTIONS": "4",
    "LLM_REFLECTION_INTERVAL": "10",
    "LLM_REFLECTION_MAX_NEW_TOKENS": "10000",
    "LLM_TRACE_IMAGES": "0",
    "LOCAL_VALIDATION_GAME_IDS": "ls20,vc33,ft09,sp80",
    "LOCAL_VALIDATION_GAME_TIME_LIMIT_S": "2400",
    "RUN_ARC_LOCAL_VALIDATION": "1",
    "VLLM_GENERATION_CONFIG": "",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.94",
    "VLLM_LIMIT_MM_PER_PROMPT": '{"image": 4}',
    "VLLM_MAX_MODEL_LEN": "32768",
    "VLLM_MAX_NUM_SEQS": "20",
    "VLLM_MODEL_PATH": "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1",
    "VLLM_QUANTIZATION": "",
}

# Internal mapping; don't edit unless Kaggle adds new options. Names must
# match Kaggle's `kaggle.accelerator` kernel-metadata values exactly.
_ACCELERATORS = {
    "cpu":     {"name": "none",             "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",    "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100",  "gpu": True},
    "rtx6000": {"name": "nvidiaRtxPro6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

_PROFILE_ENV_PLACEHOLDER = "__PROFILE_ENV_JSON__"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _with_profile_env(template: str) -> str:
    # json.dumps of a flat str->str dict is also valid Python dict syntax,
    # so the placeholder can sit directly in a `PROFILE_ENV = ...` line.
    return template.replace(_PROFILE_ENV_PLACEHOLDER, json.dumps(PROFILE_ENV))


_INSTALL_VLLM_CELL = _with_profile_env(
    dedent(
        '''\
        import os
        import subprocess
        from pathlib import Path

        PROFILE_ENV = __PROFILE_ENV_JSON__
        for key, value in PROFILE_ENV.items():
            os.environ[key] = str(value)
        print(f"ARC model profile: {os.environ['ARC_MODEL_PROFILE']}")
        print(f"vLLM model path: {os.environ['VLLM_MODEL_PATH']}")

        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["VLLM_STARTUP_TIMEOUT"] = "1000"

        extract_root = Path("/tmp/vllm_0230_offline")
        archive_candidates = [
            Path("/kaggle/input/vllm-0-23-0-tf5-wheelhouse/wheels.tar.gz"),
            Path("/kaggle/input/vllm-0-23-0-tf5/wheels.tar.gz"),
            Path("/kaggle/input/vllm-0-23-0/wheels.tar.gz"),
        ]
        archive_candidates.extend(Path("/kaggle/input").glob("**/wheels.tar.gz"))
        for archive_path in archive_candidates:
            if archive_path.exists():
                import tarfile

                extract_root.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_root)
                print(f"Extracted vLLM wheelhouse from {archive_path}")
                break

        wheel_candidates = [
            Path("/kaggle/input/datasets/ko0kip/vllm-0230-offline/vllm_0230_offline/wheels"),
            extract_root / "wheels",
            Path("/kaggle/input/vllm-deps/wheels"),
        ]
        wheel_candidates.extend(
            path for path in Path("/kaggle/input").glob("**/wheels")
            if any(path.glob("vllm*0.23.0*.whl"))
        )
        VLLM_WHEELS = next((path for path in wheel_candidates if path.exists()), None)
        if VLLM_WHEELS is None:
            raise FileNotFoundError("Could not find the offline vLLM 0.23.0 wheelhouse")

        subprocess.check_call([
            "uv", "pip", "install",
            "--no-index",
            f"--find-links={VLLM_WHEELS}",
            "vllm==0.23.0",
            "transformers==5.12.1",
        ])
        print(f"Installed vLLM from {VLLM_WHEELS}")
        '''
    )
)

_INSTALL_ARC_AGI_CELL = dedent(
    '''\
    import shutil
    import site
    import subprocess
    import sys
    from pathlib import Path

    # vLLM's wheel stack can leave an older Pillow tree behind. Remove it
    # before installing the competition wheel's Pillow 12.2.0 dependency.
    for base in site.getsitepackages():
        base_path = Path(base)
        shutil.rmtree(base_path / "PIL", ignore_errors=True)
        for dist_info in base_path.glob("pillow*dist-info"):
            shutil.rmtree(dist_info, ignore_errors=True)

    arc_wheels = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index",
        "--find-links", str(arc_wheels),
        "arc-agi",
        "python-dotenv",
    ])

    from PIL import Image
    print(f"Pillow import OK: {Image.__version__}")
    '''
)

_SMOKE_TEST_CELL = _with_profile_env(
    dedent(
        '''\
        import base64
        import io
        import importlib.util
        import os
        import shutil
        import sys
        from pathlib import Path
        from PIL import Image

        PROFILE_ENV = __PROFILE_ENV_JSON__
        for key, value in PROFILE_ENV.items():
            os.environ[key] = str(value)

        # Commit-mode guardrail: prove the final single-file agent imports
        # under Kaggle's offline framework before we risk a competition rerun.
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            framework_src = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents')
            smoke_dir = Path('/kaggle/working/ARC-AGI-3-Agents-smoke')
            if smoke_dir.exists():
                shutil.rmtree(smoke_dir)
            try:
                shutil.copytree(framework_src, smoke_dir)
                (smoke_dir / 'agents' / '__init__.py').write_text(
                    "from .agent import Agent, Playback\\n"
                    "from .swarm import Swarm\\n",
                    encoding='utf-8',
                )
                sys.path.insert(0, str(smoke_dir))

                spec = importlib.util.spec_from_file_location('my_agent_smoke', '/tmp/my_agent.py')
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                print(f'Smoke import OK: {module.MyAgent.__name__}')

                from importlib.metadata import version
                print(f"vLLM package: {version('vllm')}")
                print(f"Transformers package: {version('transformers')}")

                local_validation_requested = os.getenv('RUN_ARC_LOCAL_VALIDATION', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
                if local_validation_requested:
                    print('Skipping vLLM startup smoke; RUN_ARC_LOCAL_VALIDATION will exercise the full server path.')
                elif os.getenv('RUN_VLLM_STARTUP_SMOKE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
                    print('Starting vLLM startup smoke...')
                    try:
                        module.MyAgent._ensure_vllm_available()
                        print(f'vLLM startup smoke OK: {module.MyAgent._served_model}')
                        if os.getenv('RUN_VLLM_GENERATION_SMOKE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
                            image = Image.new('RGB', (64, 64), (0, 0, 0))
                            image.putpixel((31, 31), (255, 0, 0))
                            buffer = io.BytesIO()
                            image.save(buffer, format='PNG')
                            image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
                            smoke_n = max(1, int(os.getenv('LLM_ACTION_CANDIDATES', '1')))
                            request = {
                                'model': module.MyAgent._served_model,
                                'messages': [{
                                    'role': 'user',
                                    'content': [
                                        {'type': 'image_url', 'image_url': {'url': image_url}},
                                        {'type': 'text', 'text': 'Return JSON only: {"ok": true, "color": "red"}'},
                                    ],
                                }],
                                'max_tokens': 24,
                                'temperature': 0.2,
                                'response_format': {'type': 'json_object'},
                            }
                            if smoke_n > 1:
                                request['n'] = min(5, smoke_n)
                            smoke_response = module.MyAgent._client.chat.completions.create(**request)
                            print(f'vLLM image generation smoke choices: {len(smoke_response.choices)}')
                    finally:
                        # Kill the whole process group: orphaned engine workers keep
                        # holding GPU memory and make the real run OOM.
                        module.MyAgent._teardown_server()
            finally:
                try:
                    sys.path.remove(str(smoke_dir))
                except ValueError:
                    pass
                shutil.rmtree(smoke_dir, ignore_errors=True)
        '''
    )
)

_RUN_CELL = _with_profile_env(
    dedent(
        '''\
        from pathlib import Path
        import os
        import re
        import shutil
        import subprocess
        import sys
        import textwrap
        import time
        import urllib.error
        import urllib.request

        GATEWAY_GAMES_URL = 'http://gateway:8001/api/games'
        ARC_API_KEY = os.getenv('ARC_API_KEY') or 'test-key-123'
        PROFILE_ENV = __PROFILE_ENV_JSON__
        for key, value in PROFILE_ENV.items():
            os.environ[key] = str(value)

        def profile_env_text() -> str:
            return ''.join(f'{key}={value}\\n' for key, value in PROFILE_ENV.items())

        def gateway_available() -> bool:
            request = urllib.request.Request(
                GATEWAY_GAMES_URL,
                headers={'X-API-Key': ARC_API_KEY, 'Accept': 'application/json'},
            )
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    print(f'Gateway probe status: {response.status}')
                    return 200 <= response.status < 300
            except urllib.error.HTTPError as exc:
                print(f'Gateway probe HTTP error: {exc.code} {exc.reason}')
            except Exception as exc:
                print(f'Gateway not available in this run: {exc!r}')
            return False

        def wait_for_gateway(max_wait_sec: int) -> bool:
            deadline = time.time() + max_wait_sec
            while time.time() < deadline:
                if gateway_available():
                    return True
                time.sleep(5)
            return False

        is_competition_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
        should_run_gateway = is_competition_rerun or gateway_available()
        run_local_validation = (
            not is_competition_rerun
            and os.getenv('RUN_ARC_LOCAL_VALIDATION', '0').strip().lower()
            in {'1', 'true', 'yes', 'on'}
        )

        def prepare_framework() -> Path:
            agents_wd = Path('/kaggle/working/ARC-AGI-3-Agents')
            if agents_wd.exists():
                shutil.rmtree(agents_wd)
            shutil.copytree(
                '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',
                agents_wd,
            )
            shutil.copyfile(
                '/tmp/my_agent.py',
                agents_wd / 'agents' / 'templates' / 'my_agent.py',
            )
            (agents_wd / 'agents' / '__init__.py').write_text(
                "from typing import Type\\n"
                "from dotenv import load_dotenv\\n"
                "from .agent import Agent, Playback\\n"
                "from .swarm import Swarm\\n"
                "from .templates.random_agent import Random\\n"
                "from .templates.my_agent import MyAgent\\n\\n"
                "load_dotenv()\\n\\n"
                "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n"
                "    'random': Random,\\n"
                "    'myagent': MyAgent,\\n"
                "}\\n",
                encoding='utf-8',
            )
            return agents_wd

        def summarize_local_validation(log_path: Path) -> None:
            if not log_path.exists():
                print(f'Local validation summary unavailable; missing {log_path}')
                return
            pattern = re.compile(
                r'INFO \\| ([a-z0-9]+-[a-f0-9]+) - ([A-Z0-9]+): count (\\d+), levels completed (\\d+)'
            )
            rows = {}
            for line in log_path.read_text(errors='ignore').splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                game_id, action_name, count_raw, levels_raw = match.groups()
                count = int(count_raw)
                levels = int(levels_raw)
                row = rows.setdefault(
                    game_id,
                    {'max_levels': 0, 'last_count': 0, 'last_levels': 0, 'last_action': ''},
                )
                row['max_levels'] = max(row['max_levels'], levels)
                if count >= row['last_count']:
                    row.update(
                        last_count=count,
                        last_levels=levels,
                        last_action=action_name,
                    )
            print(
                f'LOCAL_VALIDATION games={len(rows)} total_max_levels='
                f'{sum(row["max_levels"] for row in rows.values())}'
            )
            for game_id, row in sorted(
                rows.items(), key=lambda item: (-item[1]['max_levels'], item[0])
            ):
                print(
                    f'LOCAL_VALIDATION {game_id} max={row["max_levels"]} '
                    f'last={row["last_levels"]} steps={row["last_count"]} '
                    f'last_action={row["last_action"]}'
                )

        if should_run_gateway:
            agents_wd = prepare_framework()

            # Point the framework at the gateway sidecar.
            with open(agents_wd / '.env', 'w') as f:
                f.write(
                    "SCHEME=http\\n"
                    "HOST=gateway\\n"
                    "PORT=8001\\n"
                    f"ARC_API_KEY={ARC_API_KEY}\\n"
                    "ARC_BASE_URL=http://gateway:8001/\\n"
                    "OPERATION_MODE=online\\n"
                    "ENVIRONMENTS_DIR=\\n"
                    "RECORDINGS_DIR=/kaggle/working/server_recording\\n"
                    + profile_env_text()
                )

            # Wait briefly with auth, but never hang the notebook for 10 minutes.
            # main.py will print API details if the gateway is still unavailable.
            wait_for_gateway(60 if is_competition_rerun else 10)

            # Run it. The gateway records every action and emits submission.parquet.
            run_env = os.environ.copy()
            run_env.update(PROFILE_ENV)
            run_env.update({
                'MPLBACKEND': 'agg',
                'VLLM_STARTUP_TIMEOUT': '1000',
                'VLLM_USE_FLASHINFER_SAMPLER': '0',
            })
            subprocess.run(
                [sys.executable, 'main.py', '--agent', 'myagent'],
                cwd=agents_wd,
                check=True,
                env=run_env,
            )
        elif run_local_validation:
            print('Running local validation mode')
            agents_wd = prepare_framework()
            recordings_dir = Path('/kaggle/working/server_recording')
            recordings_dir.mkdir(parents=True, exist_ok=True)
            env_dir = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
            if not env_dir.exists():
                raise FileNotFoundError(f'Local environment_files not found: {env_dir}')
            local_game_ids = [
                item.strip()
                for item in os.getenv('LOCAL_VALIDATION_GAME_IDS', '').split(',')
                if item.strip()
            ]
            if local_game_ids:
                subset_dir = Path('/kaggle/working/local_validation_environment_files')
                if subset_dir.exists():
                    shutil.rmtree(subset_dir)
                subset_dir.mkdir(parents=True, exist_ok=True)
                missing_games = []
                for game_id in local_game_ids:
                    source_dir = env_dir / game_id
                    if not source_dir.exists():
                        missing_games.append(game_id)
                        continue
                    shutil.copytree(source_dir, subset_dir / game_id)
                if missing_games:
                    raise FileNotFoundError(
                        f'Local validation games missing from environment_files: {missing_games}'
                    )
                env_dir = subset_dir
                print(f'Local validation subset: {",".join(local_game_ids)}')
            with open(agents_wd / '.env', 'w') as f:
                f.write(
                    "SCHEME=http\\n"
                    "HOST=127.0.0.1\\n"
                    "PORT=8001\\n"
                    f"ARC_API_KEY={ARC_API_KEY}\\n"
                    "ARC_BASE_URL=http://127.0.0.1:8001/\\n"
                    "OPERATION_MODE=online\\n"
                    "ENVIRONMENTS_DIR=\\n"
                    f"RECORDINGS_DIR={recordings_dir}\\n"
                    + profile_env_text()
                )
            server_log = recordings_dir / 'arc_server.log'
            server_process = subprocess.Popen(
                [
                    sys.executable,
                    '-c',
                    textwrap.dedent(
                        f"""
                        import os
                        from pathlib import Path
                        from arc_agi import Arcade, OperationMode

                        os.environ['OPERATION_MODE'] = 'OFFLINE'
                        os.environ['ENVIRONMENTS_DIR'] = {str(env_dir)!r}
                        os.environ['RECORDINGS_DIR'] = {str(recordings_dir)!r}
                        Path(os.environ['RECORDINGS_DIR']).mkdir(parents=True, exist_ok=True)
                        Arcade(
                            operation_mode=OperationMode.OFFLINE,
                            environments_dir=os.environ['ENVIRONMENTS_DIR'],
                        ).listen_and_serve(
                            host='0.0.0.0',
                            port=8001,
                            competition_mode=True,
                            save_all_recordings=True,
                        )
                        """
                    ),
                ],
                stdout=open(server_log, 'w'),
                stderr=subprocess.STDOUT,
            )
            try:
                time.sleep(5)
                env = os.environ.copy()
                env.update(PROFILE_ENV)
                env.update(
                    {
                        'MPLBACKEND': 'agg',
                        'VLLM_STARTUP_TIMEOUT': '1000',
                        'VLLM_USE_FLASHINFER_SAMPLER': '0',
                        'GAME_TIME_LIMIT_S': os.getenv('LOCAL_VALIDATION_GAME_TIME_LIMIT_S', str(60 * 60)),
                    }
                )
                subprocess.run(
                    [sys.executable, 'main.py', '--agent', 'myagent'],
                    cwd=agents_wd,
                    check=True,
                    env=env,
                )
            finally:
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait()
            summarize_local_validation(agents_wd / 'logs.log')
        else:
            print('Skipping gateway/local validation; set RUN_ARC_LOCAL_VALIDATION=1 for public-suite validation.')
        '''
    )
)

_DUMMY_SUBMISSION_CELL = dedent(
    '''\
    from pathlib import Path
    if not Path('/kaggle/working/submission.parquet').exists():
        # Save-and-run-all (commit) mode: emit a dummy submission so the
        # commit succeeds. The real submission.parquet is produced by the
        # gateway whenever it is reachable.
        import pandas as pd
        submission = pd.DataFrame(
            data=[['1_0', '1', True, 1]],
            columns=['row_id', 'game_id', 'end_of_game', 'score'])
        submission.to_parquet('/kaggle/working/submission.parquet', index=False)
        submission.head()
    '''
)


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            code_cell(_INSTALL_VLLM_CELL),
            code_cell(_INSTALL_ARC_AGI_CELL),
            write_agent_cell,
            code_cell(_SMOKE_TEST_CELL),
            code_cell(_RUN_CELL),
            code_cell(_DUMMY_SUBMISSION_CELL),
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
