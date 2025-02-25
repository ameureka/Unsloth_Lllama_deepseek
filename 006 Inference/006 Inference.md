# **Inference**

Now let's try the model we just trained! First, let's first try the model without any GRPO trained:

进行对比推理以及评测的过程

所以这里我们有一个推理对比的过程，首先是使用预训练的模型进行推理的，查看推理之后的效果，然后使用训练之后的要加载lora 的模型进行对比的看之后的效果如何。

代码使用了 `tokenizer` 对输入进行处理，并使用 `vllm` 库的 `SamplingParams` 来控制文本生成过程。最后，它利用 `unsloth` 库的 `fast_generate` 函数进行高效的文本生成，并提取最终的文本输出。

```
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

output
```

**`tokenizer.apply_chat_template(...)`**:  调用了 `tokenizer` 对象的 `apply_chat_template` 函数。  这个函数是专门为**聊天模型 (chat models)** 设计的，用于将一系列对话消息 (messages) 转换为模型能够理解和处理的输入格式。 不同的聊天模型可能对输入格式有特定的要求，`apply_chat_template` 能够根据模型的配置，自动完成格式化工作。

【所以我们就知道，如何更好的地绕过模型，特别是开源模型，那就是你知道模型底层的交流的数据的格式，这样的情况下不是明文的情况在就是可以的绕过】

**`[{"role" : "user", "content" : "Calculate pi."}]`**:  这是传递给 `apply_chat_template` 的**消息列表 (list of messages)**。 在聊天对话的上下文中，通常需要区分不同的角色，例如 "system", "user", "assistant" 等。  这里定义了一个包含**单个用户消息**的列表：

【这个就是我们的在类似的产品之中使用的，提示词模板的主要的底层的原理】

**`add_generation_prompt = True`**: 参数 `add_generation_prompt` 设置为 `True`。  很多聊天模型，特别是指令微调 (instruction-tuned) 模型，需要一个特殊的 **生成提示 (generation prompt)** 来引导模型开始生成回复。  `add_generation_prompt=True`  指示 `apply_chat_template`  **自动添加模型所需的生成提示**。  对于 Llama 3.1 Instruct 模型，通常需要添加生成提示来触发指令遵循的行为。

用于设置文本生成过程中的各种采样策略参数。

**`sampling_params = SamplingParams(...)`**:  创建 `SamplingParams` 类的**实例，并赋值给变量 `sampling_params`**。  在 `SamplingParams()` 的括号内，设置了具体的采样参数：

- **`temperature = 0.8`**: 设置**生成温度 (temperature) 为 0.8**。 `temperature` 控制模型生成文本的**随机性**。 值越高，随机性越高，多样性越丰富，但也可能降低连贯性和准确性； 值越低，随机性越低，确定性越高，更倾向于输出高概率的 token。 `0.8` 是一个常用的折衷值，旨在平衡创造性和连贯性。
- **`top_p = 0.95`**: 设置 **Top-p 采样 (nucleus sampling) 的 p 值为 0.95**。 Top-p 采样是一种**动态 token 选择策略**，它会从累积概率达到 `p` 值的候选 token 集合中进行采样。 `top_p` 值越高，采样范围越大，多样性越高，但也可能引入更多低质量的 token。 `0.95` 是一个常用的 Top-p 值，旨在保证生成文本质量的同时，保持一定的多样性。
- **`max_tokens = 1024`**: 设置**模型生成回复的最大 token 数量 (maximum tokens) 为 1024**。 `max_tokens` 限制了模型单次生成的文本长度，防止模型生成过长的回复，超出上下文窗口或计算资源限制。 `1024` token 通常对于大多数文本生成任务来说已经足够长。

【所以这里基本了解了大模型的底层的输入与输出的原理，对于开源前端应用来说是个比较好的状态的，或者说一个比较好的策略】

这行代码的作用是： **调用 `model.fast_generate` 函数，使用之前格式化的输入文本 `text` 和设置好的采样参数 `sampling_params`，利用 vLLM 进行高效的文本生成。 然后从 `fast_generate` 的返回结果中，提取模型生成的文本回复字符串，并赋值给变量 `output`。**

结果：

<img width="713" alt="Clipboard_Screenshot_1740497372" src="https://github.com/user-attachments/assets/883062c9-66c2-49af-a2c1-24e5d7c9d680" />


**Calculating Pi using the Bailey-Borwein-Plouffe Formula**

Pi (π) is an irrational number that represents the ratio of a circle's circumference to its diameter. It's an essential constant in mathematics and appears in many mathematical and scientific applications. One way to calculate pi is using the Bailey-Borwein-Plouffe (BBP) formula, a spigot algorithm for computing the nth binary digit of pi.

**The BBP Formula:**

The BBP formula is given by:

π = Σ (1 / (16^k)) * ((4 / (8k + 1)) + (2 / (8k + 4)) + (1 / (8k + 5)) + (1 / (8k + 6)) - (1 / (8k + 3)) - (1 / (8k + 5)) - (1 / (8k + 2)) - (1 / (8k + 7))) (k = 0 to infinity)

**Implementation in Python:**

```python
def calculate_pi(n):
    pi = 0.0
    for k in range(n):
        pi += 1 / (16 ** k) * (
            (4 / (8 * k + 1)) +
            (2 / (8 * k + 4)) +
            (1 / (8 * k + 5)) +
            (1 / (8 * k + 6)) -
            (1 / (8 * k + 3)) -
            (1 / (8 * k + 5)) -
            (1 / (8 * k + 2)) -
            (1 / (8k + 7))
        )
    return pi

# Calculate pi up to 50 iterations
pi_value = calculate_pi(50)
print("Approximate value of pi:", pi_value)
```

**Note:** The BBP formula converges slowly and may not be the most efficient way to calculate pi, especially for high precision. For more accurate and efficient methods, consider using specialized libraries or algorithms like the Gauss-Legendre algorithm or the Chudnovsky algorithm.

**Limitations:**

- The BBP formula is an approximation and may not provide exact

以上内容是转换成markdown格式之后内容形似。

```
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

output
```

那么我们就需要详细分析下当前的输入与后续的输入的之间的差别是什么，如何进行。

 {"role" : "user", "content" : "Calculate pi."},

**增加了系统消息**:  当前版本中，输入消息列表 **在用户消息之前，增加了一个系统消息**：

- `{"role" : "system", "content" : SYSTEM_PROMPT}`： 这个消息的角色是 "system"，内容是 `SYSTEM_PROMPT` 变量的值。
- **`SYSTEM_PROMPT` 的意义**: `SYSTEM_PROMPT` 变量在之前的代码中被定义为 **系统提示词**，很可能包含了 **指导模型生成特定格式或风格回复的指令** (例如 XML-CoT 格式)。

**预期输出格式**:  由于使用了 `SYSTEM_PROMPT`，我们可以 **预期模型生成的回复会受到系统提示词的影响**，例如，如果 `SYSTEM_PROMPT`  指示模型使用 XML-CoT 格式回复，那么我们期望新版本生成的回复 **更有可能符合 XML-CoT 格式**。

**期望性能提升**:  由于模型已经通过 GRPO 进行了微调，我们 **期望新版本生成的回复在质量和符合训练目标方面都有所提升**。  如果 GRPO 训练的目标是提高模型在数学问题解决方面的能力，并使其生成 XML-CoT 格式的回复，那么我们 **预期新版本模型在回答 "Calculate pi."  时，会给出更准确、更符合 XML-CoT 格式的回复** (虽然 “Calculate pi.” 这个问题本身可能不太适合 XML-CoT 格式，因为不需要复杂的推理过程)。

结果：

reasoning
To calculate pi, we can use the Leibniz formula, which states that pi can be approximated using the infinite series:

pi/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...

This series is a simple and effective way to approximate pi, but it is an infinite series, so it will take a long time to compute the result exactly.

However, we can use a finite number of terms to get an approximation of pi. Let's use 1000 terms to calculate pi.

We will use the following Python code to calculate the approximation of pi:

```python
def calculate_pi(n):
    pi = 0.0
    for i in range(n):
        pi += ((-1)**i) / (2*i + 1)
    return 4 * pi

n = 1000
pi = calculate_pi(n)
print("Approximation of pi:", pi)

```

This code will output an approximation of pi using 1000 terms of the series.

However, a more efficient and accurate method to calculate pi is to use the Gauss-Legendre algorithm, which is a more complex algorithm that uses a recursive formula to calculate pi.

But, for simplicity and ease of calculation, the above code will do.

answer
The approximation of pi is 3.1415926536
Note that the actual value of pi is an irrational number and has many digits after the decimal point, so the above approximation is only an approximation.

我们发现整个的过程包含了思维链的推导的过程。



