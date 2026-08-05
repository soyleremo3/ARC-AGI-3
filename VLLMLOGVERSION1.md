(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339] 
(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339]        █     █     █▄   ▄█
(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.23.0
(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339]   █▄█▀ █     █     █     █  model   /kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1
(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=270) INFO 08-04 13:08:12 [api_utils.py:339] 
(APIServer pid=270) INFO 08-04 13:08:13 [api_utils.py:273] non-default args: {'host': '127.0.0.1', 'model': '/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1', 'trust_remote_code': True, 'max_model_len': 32768, 'served_model_name': ['vllm-model'], 'gpu_memory_utilization': 0.94, 'enable_prefix_caching': True, 'limit_mm_per_prompt': {'image': 4}, 'max_num_seqs': 20}
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_GPU_MEMORY_UTILIZATION
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_MODEL_PATH
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_MAX_MODEL_LEN
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_GENERATION_CONFIG
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_MAX_NUM_SEQS
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_LIMIT_MM_PER_PROMPT
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_QUANTIZATION
(APIServer pid=270) WARNING 08-04 13:08:13 [envs.py:2088] Unknown vLLM environment variable detected: VLLM_STARTUP_TIMEOUT
(APIServer pid=270) INFO 08-04 13:08:22 [model.py:611] Resolved architecture: Gemma4ForConditionalGeneration
(APIServer pid=270) INFO 08-04 13:08:22 [model.py:1745] Using max model len 32768
(APIServer pid=270) INFO 08-04 13:08:25 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=8192.
(APIServer pid=270) INFO 08-04 13:08:25 [config.py:100] Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.
(APIServer pid=270) INFO 08-04 13:08:25 [vllm.py:999] Asynchronous scheduling is enabled.
(APIServer pid=270) INFO 08-04 13:08:25 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(APIServer pid=270) WARNING 08-04 13:08:25 [cuda.py:243] Forcing --disable_chunked_mm_input for models with multimodal-bidirectional attention.
(EngineCore pid=594) INFO 08-04 13:08:34 [core.py:113] Initializing a V1 LLM engine (v0.23.0) with config: model='/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1', speculative_config=None, tokenizer='/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=vllm-model, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::qwen_gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 40, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=594) INFO 08-04 13:08:36 [parallel_state.py:1568] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.19.2.2:56735 backend=nccl
[W804 13:08:36.623518162 socket.cpp:207] [c10d] The hostname of the client socket cannot be retrieved. err=-3
(EngineCore pid=594) INFO 08-04 13:08:36 [parallel_state.py:1903] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=594) INFO 08-04 13:08:37 [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.
(EngineCore pid=594) INFO 08-04 13:08:37 [gpu_model_runner.py:5092] Starting to load model /kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1...
(EngineCore pid=594) INFO 08-04 13:08:37 [vllm.py:999] Asynchronous scheduling is enabled.
(EngineCore pid=594) INFO 08-04 13:08:37 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(EngineCore pid=594) INFO 08-04 13:08:37 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(EngineCore pid=594) INFO 08-04 13:08:37 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(EngineCore pid=594) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
(EngineCore pid=594) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
(EngineCore pid=594) INFO 08-04 13:08:42 [weight_utils.py:922] Filesystem type for checkpoints: NFS. Checkpoint size: 58.25 GiB. Available RAM: 171.16 GiB.
(EngineCore pid=594) INFO 08-04 13:08:42 [weight_utils.py:884] Prefetching checkpoint files into page cache started (in background, num_threads=8, block_size=16777216 bytes)
(EngineCore pid=594) 
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
(EngineCore pid=594) INFO 08-04 13:09:54 [weight_utils.py:856] Prefetching checkpoint files: 10% (1/2)
(EngineCore pid=594) INFO 08-04 13:15:49 [weight_utils.py:856] Prefetching checkpoint files: 20% (2/2)
(EngineCore pid=594) 
Loading safetensors checkpoint shards:  50% Completed | 1/2 [07:06<07:06, 426.74s/it]
(EngineCore pid=594) INFO 08-04 13:15:49 [weight_utils.py:879] Prefetching checkpoint files into page cache finished in 426.74s
(EngineCore pid=594) 
Loading safetensors checkpoint shards: 100% Completed | 2/2 [07:08<00:00, 176.50s/it]
(EngineCore pid=594) 
Loading safetensors checkpoint shards: 100% Completed | 2/2 [07:08<00:00, 214.04s/it]
(EngineCore pid=594) 
(EngineCore pid=594) INFO 08-04 13:15:50 [default_loader.py:397] Loading weights took 428.12 seconds
(EngineCore pid=594) INFO 08-04 13:15:51 [gpu_model_runner.py:5187] Model loading took 58.99 GiB memory and 433.184810 seconds
(EngineCore pid=594) INFO 08-04 13:15:51 [gpu_model_runner.py:6200] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 3 video items of the maximum feature size.
(EngineCore pid=594) INFO 08-04 13:16:13 [backends.py:1089] Using cache directory: /root/.cache/vllm/torch_compile_cache/2153aa71e0/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=594) INFO 08-04 13:16:13 [backends.py:1148] Dynamo bytecode transform time: 7.96 s
(EngineCore pid=594) INFO 08-04 13:16:21 [backends.py:378] Cache the graph of compile range (1, 8192) for later use
(EngineCore pid=594) INFO 08-04 13:16:32 [backends.py:393] Compiling a graph for compile range (1, 8192) takes 17.83 s
(EngineCore pid=594) INFO 08-04 13:16:36 [decorators.py:708] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/1e2a666b8ddb17898c36d6327fc888fe909b288c8777c4357fab8adf1e9f2922/rank_0_0/model
(EngineCore pid=594) INFO 08-04 13:16:36 [monitor.py:53] torch.compile took 30.08 s in total
(EngineCore pid=594) INFO 08-04 13:16:37 [monitor.py:81] Initial profiling/warmup run took 1.25 s
(EngineCore pid=594) INFO 08-04 13:16:42 [gpu_model_runner.py:6412] Profiling CUDA graph memory: PIECEWISE=8 (largest=40), FULL=5 (largest=16)
(EngineCore pid=594) INFO 08-04 13:16:47 [gpu_model_runner.py:6517] Estimated CUDA graph memory: 0.13 GiB total
(EngineCore pid=594) INFO 08-04 13:16:47 [gpu_worker.py:480] Available KV cache memory: 28.41 GiB
(EngineCore pid=594) INFO 08-04 13:16:47 [gpu_worker.py:495] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9400 is equivalent to --gpu-memory-utilization=0.9386 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9414. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=594) INFO 08-04 13:16:47 [kv_cache_utils.py:1744] GPU KV cache size: 97,541 tokens
(EngineCore pid=594) INFO 08-04 13:16:47 [kv_cache_utils.py:1745] Maximum concurrency for 32,768 tokens per request: 2.98x
(EngineCore pid=594) 2026-08-04 13:16:47,790 - INFO - autotuner.py:622 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(EngineCore pid=594) 2026-08-04 13:16:47,802 - INFO - autotuner.py:641 - flashinfer.jit: [Autotuner]: Autotuning process ends
(EngineCore pid=594) 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/8 [00:00<?, ?it/s]
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  25%|██▌       | 2/8 [00:00<00:00, 15.07it/s]
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  50%|█████     | 4/8 [00:00<00:00, 15.55it/s]
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  75%|███████▌  | 6/8 [00:00<00:00, 15.80it/s]
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 8/8 [00:00<00:00, 15.07it/s]
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 8/8 [00:00<00:00, 15.24it/s]
(EngineCore pid=594) 
Capturing CUDA graphs (decode, FULL):   0%|          | 0/5 [00:00<?, ?it/s]
Capturing CUDA graphs (decode, FULL):  40%|████      | 2/5 [00:00<00:00, 17.21it/s]
Capturing CUDA graphs (decode, FULL):  80%|████████  | 4/5 [00:00<00:00, 17.42it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████| 5/5 [00:01<00:00,  3.87it/s]
(EngineCore pid=594) INFO 08-04 13:16:51 [gpu_model_runner.py:6585] Graph capturing finished in 4 secs, took 0.10 GiB
(EngineCore pid=594) INFO 08-04 13:16:51 [gpu_worker.py:639] CUDA graph pool memory: 0.1 GiB (actual), 0.13 GiB (estimated), difference: 0.03 GiB (26.9%).
(EngineCore pid=594) INFO 08-04 13:16:51 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(EngineCore pid=594) INFO 08-04 13:16:52 [core.py:306] init engine (profile, create kv cache, warmup model) took 60.86 s (compilation: 30.08 s)
(EngineCore pid=594) INFO 08-04 13:16:52 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(APIServer pid=270) INFO 08-04 13:16:52 [api_server.py:579] Supported tasks: ['generate']
(APIServer pid=270) WARNING 08-04 13:16:52 [model.py:1502] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=270) INFO 08-04 13:16:53 [hf.py:548] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=270) INFO 08-04 13:17:26 [base.py:227] Multi-modal warmup completed in 33.069s
(APIServer pid=270) INFO 08-04 13:17:27 [base.py:227] Readonly multi-modal warmup completed in 0.063s
(APIServer pid=270) INFO 08-04 13:17:27 [api_server.py:583] Starting vLLM server on http://127.0.0.1:8000
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:37] Available routes are:
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /docs, Methods: GET, HEAD
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /generative_scoring, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=270) INFO 08-04 13:17:27 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=270) INFO:     Started server process [270]
(APIServer pid=270) INFO:     Waiting for application startup.
(APIServer pid=270) INFO:     Application startup complete.
(APIServer pid=270) INFO:     127.0.0.1:56656 - "GET /v1/models HTTP/1.1" 200 OK
(EngineCore pid=594) WARNING 08-04 13:17:28 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=594) WARNING 08-04 13:17:28 [jit_monitor.py:103] Triton kernel JIT compilation during inference: kernel_unified_attention. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=594) WARNING 08-04 13:17:29 [jit_monitor.py:103] Triton kernel JIT compilation during inference: apply_token_bitmask_inplace_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(APIServer pid=270) INFO:     127.0.0.1:56656 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=270) INFO 08-04 13:17:30 [launcher.py:100] [shutdown] API server: shutdown triggered
(APIServer pid=270) INFO 08-04 13:17:30 [launcher.py:116] [shutdown] API server: stopping engine client mode=abort timeout=0s
(APIServer pid=270) INFO 08-04 13:17:30 [core_client.py:652] [shutdown] MPClient: start timeout=0s
(APIServer pid=270) INFO 08-04 13:17:30 [core_client.py:654] [shutdown] MPClient: stopping engine manager
(APIServer pid=270) WARNING 08-04 13:17:30 [utils.py:607] [shutdown] Process manager: force killing remaining processes count=1
(APIServer pid=270) INFO 08-04 13:17:30 [core_client.py:656] [shutdown] MPClient: engine manager stopped
(APIServer pid=270) INFO 08-04 13:17:30 [core_client.py:657] [shutdown] MPClient: cleaning up background resources
(APIServer pid=270) INFO 08-04 13:17:30 [core_client.py:659] [shutdown] MPClient: complete
(APIServer pid=270) INFO 08-04 13:17:30 [launcher.py:125] [shutdown] API server: engine client stopped
(APIServer pid=270) INFO 08-04 13:17:30 [launcher.py:128] [shutdown] API server: signalling HTTP server shutdown
(APIServer pid=270) INFO 08-04 13:17:30 [launcher.py:149] [shutdown] API server: shutting down FastAPI HTTP server
(APIServer pid=270) INFO:     Shutting down
(APIServer pid=270) INFO:     Waiting for application shutdown.
(APIServer pid=270) INFO:     Application shutdown complete.
/usr/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '