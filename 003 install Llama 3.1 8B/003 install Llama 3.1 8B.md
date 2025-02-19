Load up `Llama 3.1 8B Instruct`, and set parameters

这段代码主要负责**加载预训练的 Llama 3.1 8B Instruct 模型**，并对其进行**配置**，以便后续可以进行推理或微调。 代码中使用了 `unsloth` 库来高效地加载和配置模型，并应用了 **LoRA (Low-Rank Adaptation)** 技术来为后续的微调做准备。 让我们逐行分解这段代码：

【ps 模型的微调技术的lora】

```
from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
```

`bfloat16` 是一种用于深度学习的半精度浮点格式，相比传统的 `float32` (单精度浮点) 可以**减少内存占用并加速计算**，尤其是在支持它的硬件（例如某些较新的 NVIDIA GPU）上。 `unsloth` 库可能会根据硬件是否支持 `bfloat16` 来进行优化。

**`import torch`**: 导入 PyTorch 库。 PyTorch 是一个广泛使用的深度学习框架，`unsloth` 库很可能基于 PyTorch 构建。

`max_seq_length` 可以让模型处理更长的文本输入，从而在推理时可以考虑更长的上下文信息，实现更长的 "推理轨迹" (即模型可以记住并利用更长的对话历史或文档内容)。  但是，增加 `max_seq_length` 通常会**增加计算资源和内存需求**。

**`max_seq_length`**:  表示**模型能够处理的最大序列长度**。 在处理文本时，文本会被切分成一个个 token (令牌)，序列长度就是指输入文本的 token 数量。

**`lora_rank`**:  这是 **LoRA (Low-Rank Adaptation) 的秩 (rank)** 参数。 LoRA 是一种参数高效的微调技术，它通过在预训练模型中注入少量可训练的低秩矩阵，来实现模型的适配，而无需微调整个模型的所有参数。 `rank` 是 LoRA 的一个关键参数，它决定了注入的低秩矩阵的维度。

更大的 `rank` 也会导致计算量增加，推理速度可能会变慢，并且需要更多的 GPU 内存。  `32` 是一个相对常用的 `lora_rank` 值，通常在性能和效率之间取得较好的平衡。

【告诉的大家的，这个模型在进行Lora 训练的时候，使用的什么样的训练的参数】

```jsx
model, tokenizer = FastLanguageModel.from_pretrained(
model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
max_seq_length = max_seq_length,
load_in_4bit = True, # LoRA 16bit时为False
fast_inference = True, # 启用vLLM快速推理
max_lora_rank = lora_rank,
gpu_memory_utilization = 0.6, # 如果内存不足则减少
)
```

FastLanguageModel.from_pretrained 它返回两个对象： `model` (加载的模型) 和 `tokenizer` (用于文本 tokenization 的 tokenizer)。

meta-llama/meta-Llama-3.1-8B-Instruct 这是一个 80 亿参数的语言模型，并且是经过指令微调 (Instruct) 的版本，更擅长理解和遵循人类指令。

**`load_in_4bit = True`**:  启用 **4-bit 量化加载**模型。 这是一种 **模型压缩技术**，可以将模型的权重从通常的 16-bit 或 32-bit 浮点数 **量化到 4-bit 整数**。  这样做可以 **大幅度减少模型占用的 GPU 内存**，使得在 GPU 资源有限的情况下也能运行大型模型。

**`# 如果内存不足则减少`**:  注释说明，如果 **GPU 内存不足** (例如运行代码时出现 Out Of Memory 错误)，可以 **减少** `gpu_memory_utilization` 的值，例如降低到 `0.5` 或更低。  降低内存利用率可以减少模型占用的显存，但可能会 **限制模型的性能或最大序列长度**。

【以上一点是在大模型的微调过程之中的技巧，就是控制模型占比gpu 的大小】

**`tokenizer`**:  是与 Llama 3.1 8B Instruct 模型配套使用的 **tokenizer 对象**，用于将文本转换为模型可以理解的 token 序列，以及将模型输出的 token 序列转换回文本。

【我们要注意一点就是的，什么是的模型可以理解的token 序列】

结果：

这段代码输出信息表明你成功使用 Unsloth 库加载了 `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit` 模型，并进行了 4-bit 量化。环境配置为 Tesla T4 GPU, PyTorch 2.5.1, CUDA Toolkit 12.4, Triton 3.1.0。模型加载时间约为 5.35 秒，整个引擎初始化时间约为 54.71 秒。模型权重占用约 5.35 GB 显存，KV 缓存预留了 2.64 GB。 虽然有数据类型转换的警告，但 XFormers 和 CUDA Graph 等优化技术都被启用，以提升推理性能。 总体而言，模型加载和初始化过程顺利完成，可以开始进行模型推理任务。

```jsx
 Unsloth 2025.2.5: Fast Llama patching. Transformers: 4.48.2.
\\   /|    GPU: Tesla T4. Max memory: 14.741 GB. Platform: Linux.
O^O/ \*/ \    Torch: 2.5.1+cu124. CUDA: 7.5. CUDA Toolkit: 12.4. Triton: 3.1.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.28.post3. FA2 = False]
"-*___-"     Free Apache license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/meta-llama-3.1-8b-instruct-bnb-4bit with actual GPU utilization = 59.59%
Unsloth: Your GPU has CUDA compute capability 7.5 with VRAM = 14.74 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 512. Num Sequences = 160.
Unsloth: vLLM's KV Cache can use up to 2.61 GB. Also swap space = 2 GB.
WARNING 02-13 15:54:15 [config.py:2386](http://config.py:2386/)] Casting torch.bfloat16 to torch.float16.
INFO 02-13 15:54:28 [config.py:542](http://config.py:542/)] This model supports multiple tasks: {'reward', 'score', 'embed', 'classify', 'generate'}. Defaulting to 'generate'.
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'float16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection'], 'llm_int8_threshold': 6.0}
INFO 02-13 15:54:28 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='unsloth/meta-llama-3.1-8b-instruct-bnb-4bit', speculative_config=None, tokenizer='unsloth/meta-llama-3.1-8b-instruct-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=512, download_dir=None, load_format=LoadFormat.BITSANDBYTES, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/meta-llama-3.1-8b-instruct-bnb-4bit, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":0,"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":160}, use_cached_outputs=False,
tokenizer_config.json: 100%
 55.5k/55.5k [00:00<00:00, 3.88MB/s]
tokenizer.json: 100%
 17.2M/17.2M [00:00<00:00, 40.0MB/s]
special_tokens_map.json: 100%
 454/454 [00:00<00:00, 31.7kB/s]
generation_config.json: 100%
 239/239 [00:00<00:00, 14.4kB/s]
INFO 02-13 15:54:31 [cuda.py:179](http://cuda.py:179/)] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 02-13 15:54:31 [cuda.py:227](http://cuda.py:227/)] Using XFormers backend.
INFO 02-13 15:54:32 model_runner.py:1110] Starting to load model unsloth/meta-llama-3.1-8b-instruct-bnb-4bit...
INFO 02-13 15:54:32 [loader.py:1102](http://loader.py:1102/)] Loading weights with BitsAndBytes quantization.  May take a while ...
INFO 02-13 15:54:33 weight_utils.py:252] Using model weights format ['*.safetensors']
model.safetensors: 100%
 5.70G/5.70G [00:47<00:00, 190MB/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:23<00:00, 23.68s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:23<00:00, 23.09s/it]
INFO 02-13 15:56:09 model_runner.py:1115] Loading model weights took 5.3541 GB
INFO 02-13 15:56:09 punica_selector.py:18] Using PunicaWrapperGPU.
INFO 02-13 15:56:20 [worker.py:267](http://worker.py:267/)] Memory profiling takes 11.34 seconds
INFO 02-13 15:56:20 [worker.py:267](http://worker.py:267/)] the current vLLM instance can use total_gpu_memory (14.74GiB) x gpu_memory_utilization (0.60) = 8.78GiB
INFO 02-13 15:56:20 [worker.py:267](http://worker.py:267/)] model weights take 5.35GiB; non_torch_memory takes 0.05GiB; PyTorch activation peak memory takes 0.74GiB; the rest of the memory reserved for KV Cache is 2.64GiB.
INFO 02-13 15:56:21 executor_base.py:110] # CUDA blocks: 1353, # CPU blocks: 1024
INFO 02-13 15:56:21 executor_base.py:115] Maximum concurrency for 512 tokens per request: 42.28x
INFO 02-13 15:56:22 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing gpu_memory_utilization or switching to eager mode. You can also reduce the max_num_seqs as needed to decrease memory usage.Capturing CUDA graph shapes: 100%|██████████| 23/23 [00:40<00:00,  1.78s/it]INFO 02-13 15:57:03 model_runner.py:1562] Graph capturing finished in 41 secs, took 0.58 GiB
INFO 02-13 15:57:03 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 54.71 secondstokenizer_config.json: 100%
 55.5k/55.5k [00:00<00:00, 4.27MB/s]
tokenizer.json: 100%
 17.2M/17.2M [00:00<00:00, 44.4MB/s]
special_tokens_map.json: 100%
 454/454 [00:00<00:00, 37.9kB/s]
```

**配置信息 (Configuration):**

- **Bfloat16:** `FALSE` *(未使用 bfloat16 数据类型)*
- **FA [Xformers = 0.0.28.post3. FA2 = False]:** *(使用了 XFormers 优化库版本 0.0.28.post3，但未使用 FlashAttention-2)*
- **Quantization:** 4-bit Bitsandbytes (`load_in_4bit: True`, `bnb_4bit_quant_type: nf4`, `bnb_4bit_use_double_quant: True`) *(模型使用了 4-bit Bitsandbytes 量化，类型为 NF4，并使用了双重量化)*
- **Compute Data Type:** `bnb_4bit_compute_dtype: 'float16'` *(4-bit 量化计算时使用 float16 数据类型)*
- **Storage Data Type:** `bnb_4bit_quant_storage: 'uint8'` *(4-bit 量化存储时使用 uint8 数据类型)*
- **Data Type Casting:** `Casting torch.bfloat16 to torch.float16` *(由于环境配置，程序将 bfloat16 数据类型转换为 float16)*
- **Task Support:** `This model supports multiple tasks: {'reward', 'score', 'embed', 'classify', 'generate'}. Defaulting to 'generate'.` *(模型支持多种任务，默认设置为 'generate' 文本生成任务)*
- **Max Sequence Length:** `max_seq_len=512` *(模型支持的最大序列长度为 512 tokens)*
- **Chunked Prefill Tokens:** `Chunked prefill tokens = 512` *(分块预填充 tokens 数量为 512)*
- **Number of Sequences:** `Num Sequences = 160` *(最大并发序列数为 160)*
- **Conservativeness:** `Using conservativeness = 1.0` *(保守性设置为 1.0，可能与内存管理有关)*
- **KV Cache Memory:** `vLLM's KV Cache can use up to 2.61 GB. Also swap space = 2 GB.` *(vLLM 的 KV 缓存最大可以使用 2.61 GB 显存，并预留了 2GB 的交换空间)*
- **XFormers Backend:** `Using XFormers backend.` *(使用 XFormers 后端进行加速)*
- **CUDA Graph Capturing:** `Capturing cudagraphs for decoding.` *(启用了 CUDA Graph 捕获以加速解码过程)*

**模型加载信息 (Model Loading Information):**

- **Model Name:** `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit` *(加载的模型名称为 unsloth/meta-llama-3.1-8b-instruct-bnb-4bit)*
- **Model Weights Format:** `Using model weights format ['*.safetensors']` *(模型权重文件格式为 safetensors)*
- **Model Weights Size:** `model weights take 5.35GiB` *(模型权重占用约 5.35 GB 显存)*
- **Model Weights Loading Time:** `Loading model weights took 5.3541 GB` *(加载模型权重耗时约 5.35 秒)*
- **Total Model Loading Time (Engine Init):** `init engine (profile, create kv cache, warmup model) took 54.71 seconds` *(整个模型引擎初始化（包括 profile, 创建 KV 缓存, 模型预热）耗时约 54.71 秒)*
- **CUDA Graph Capturing Time:** `Graph capturing finished in 41 secs, took 0.58 GiB` *(CUDA Graph 捕获耗时 41 秒，额外占用 0.58 GB 显存)*

```
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
```

**`model = FastLanguageModel.get_peft_model(`**:  调用 `unsloth` 库的 `FastLanguageModel.get_peft_model()` 函数，来应用 **PEFT (Parameter-Efficient Fine-Tuning) 技术**，具体来说是 **LoRA**。  `get_peft_model()` 函数的作用是在已加载的模型上 **添加 LoRA 适配器**，为后续的参数高效微调做准备。  函数返回应用了 LoRA 适配器后的新模型对象，并重新赋值给 `model` 变量，替换掉之前仅加载的原始模型。

【所以这里我们就发现的，现在大家经常说PEFT的概念或者说意义是什么】

**`选择任意大于0的数字！建议8, 16, 32, 64, 128`**:  注释提示，`r` 可以选择 **任意大于 0 的数字**，并建议使用 **8, 16, 32, 64, 128** 这些值。 这些是常用的 LoRA rank 值，可以根据具体需求和资源情况进行选择。  较小的 `rank` 节省资源，较大的 `rank` 可能模型更具表现力。

`"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"`。  这些都是 **Transformer 模型架构中常见的线性层名称**。  具体来说， `"q_proj", "k_proj", "v_proj", "o_proj"`  通常是 **自注意力机制 (Self-Attention)** 中的 **Query, Key, Value 投影层和输出投影层** (QKVO)。

`"gate_proj", "up_proj", "down_proj"`  通常是 **MLP (多层感知机) 或 Feed-Forward Network (前馈网络)** 中的 **门控机制和中间层投影层** (在某些 Transformer 变体中)。  这些层是 Transformer 模型中参数量较大且比较重要的线性变换层。 对这些层应用 LoRA 可以有效地微调模型，同时保持参数高效性。

【所以你如果要高效理解，transform 机制可能要知道，这些】

是 LoRA 的 **Alpha 参数**。  它用于 **调整 LoRA 适配器对原始模型的影响程度**。  通常将 `lora_alpha` 设置为等于 `lora_rank` 或其两倍的值 (例如 `2 * lora_rank`)。  `lora_alpha` 的具体作用与 LoRA 的实现细节有关，但一般而言，更大的 `lora_alpha` 值可能会 **增大 LoRA 的影响，并可能提高模型微调的幅度**。

【微调经验，现在知道了吧】

**梯度检查点 (Gradient Checkpointing)** 技术，并指定使用 `unsloth` 库提供的实现。 梯度检查点是一种 **内存优化技术**，尤其在训练 **大型模型和长序列** 时非常有用。  它可以 **显著减少训练过程中 GPU 内存的峰值占用**，使得在有限的 GPU 资源下可以训练更大的模型或更长的序列。  其原理是在前向传播时只保存部分层的激活值，在反向传播时，需要重新计算那些没有保存的层的激活值。 这样做的代价是 **略微增加计算时间** (需要重新计算激活值)，但可以 **大幅度节省内存**。

【深入理解在，什么叫做向后微调，以及向前传播的，这些技术的区别】

设置随机种子可以 **保证实验的可复现性**。  在深度学习中，很多过程涉及到随机性，例如参数初始化、数据 shuffle 等。  设置相同的随机种子，可以确保在多次运行代码时，这些随机过程的初始状态和行为都是相同的，从而得到 **一致的结果**。

结果： 

***Unsloth 2025.2.5 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.***



