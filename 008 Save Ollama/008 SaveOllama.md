# **GGUF / llama.cpp Conversion**

**GGUF / llama.cpp Conversion**

To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):

- `q8_0` - Fast conversion. High resource use, but generally acceptable.
- `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
- `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

[**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)

分析下是什么意思，关于大模型文件的保存以及保留的问题

模型转换为 GGUF 格式，以便与 `llama.cpp` 工具库配合使用。  `llama.cpp` 是一个用于在 CPU 和 GPU 上高效运行大型语言模型的库，而 GGUF (Globally Guiding Unification Format) 是 `llama.cpp` 项目中使用的一种模型文件格式，旨在提高效率和兼容性。

**克隆 llama.cpp**:  工具库在进行 GGUF 转换过程中，会在后台 **克隆 `llama.cpp` 仓库**。  `llama.cpp`  本身包含了一些模型转换和量化的工具，因此克隆仓库可能是为了利用这些工具。  这通常是自动化操作，用户无需手动干预。

- **`q8_0`**: 是一种 **8-bit 量化方法**。 "q" 代表量化 (quantization)， "8" 代表使用 8 比特来表示权重， "0" 可能代表某种特定的量化配置或参数 (需要更详细的文档才能确认 "0" 的具体含义)。
- **意义**: `q8_0` 作为默认选项，可能 **在速度和模型大小之间取得了较好的平衡**，对于大多数用户来说是一个合理的默认选择。 8-bit 量化可以在一定程度上减小模型大小，并提升推理速度，同时通常能保持相对较好的模型性能。

【所以从这里我们是可以看到。其实在llama 拉取的模型其实也是进行压缩的，使用gguf 进行了转换】

**`q4_k_m`**:  是一种 **4-bit 量化方法**，  通常比 8-bit 量化 **更激进**，可以进一步减小模型大小和提高速度，但可能也会 **带来更大的精度损失**。  `k_m`  可能代表 "K-quants mixed precision" 或类似的含义，暗示这种方法可能在不同层或张量上使用不同的量化策略。

- **`save_pretrained_gguf`**: 用于 **将模型以 GGUF 格式保存到本地磁盘**。 类似于之前的 `save_pretrained_merged` 和 `save_pretrained_lora`，但这次是专门用于 GGUF 格式。
- **`push_to_hub_gguf`**: 用于 **将 GGUF 格式的模型推送到 Hugging Face Hub**。 同样类似于 `push_to_hub_merged` 和 `push_to_hub_lora`，但针对 GGUF 格式。
- **意义**: **提供了便捷的 API**。 用户可以使用这两个专门的函数，轻松完成 GGUF 格式模型的本地保存和云端分享。
- **Ollama notebook**: 提到了一个 **Ollama notebook**，用于 **微调模型并自动导出到 Ollama**。 Ollama 是一个用于打包、分发和运行大型语言模型的工具。
- **意义**: **扩展了工具库的功能**，使其不仅支持 GGUF/llama.cpp，还 **与 Ollama 集成**，提供了更完整的模型微调、导出和部署的解决方案。 Ollama notebook 可能是预先配置好的 Jupyter Notebook，包含了使用 Ollama 进行模型微调和导出的代码示例和步骤指导。

```
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )
```

**总结 "保存为多种 GGUF 选项" 部分:**

- **操作**: **一次性推送多种量化格式的 GGUF 模型到 Hugging Face Hub** (`q4_k_m`, `q8_0`, `q5_k_m`)。 本地保存 ( `save_pretrained_gguf` ) **没有提供一次保存多种格式的选项**，这个特性只在 `push_to_hub_gguf` 中提供。
- **特点**: **高效地批量生成和推送多种量化格式的 GGUF 模型**。 方便为用户提供不同大小和性能的模型选择。
- **用户需要做的**: 替换 Hugging Face Hub 仓库名和访问令牌 (如果需要推送)。 根据需要修改 `quantization_method` 列表，添加或删除量化方法。



