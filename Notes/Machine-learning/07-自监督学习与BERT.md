# 听课笔记7：自监督学习与 BERT

## 一、自监督学习（Self-supervised Learning）

- 定义：自己想办法在没有标注的情况下做监督——一部分输入用作模型的输入，另一部分用作模型的输出（伪标签）。
- 代表模型：ELMo、BERT、ERNIE、Big Bird

### 模型规模对比

| 模型 | 参数量 |
| --- | --- |
| ELMo | 94M |
| BERT | 340M |
| GPT-2 | 1500M (1.5B) |
| Megatron | 8B |
| T5 | 11B |
| Turing NLG | 17B |
| GPT-3 | 175B |

### 相关模型概念补齐

- **ELMo**（2018）：基于双向 LSTM 的语言模型，首次让词向量带上上下文信息；但只作为特征提取器（feature-based），不做微调。
- **BERT**（2018）：Transformer 的 Encoder，双向，用 MLM + NSP 预训练，微调迁移（本笔记主体）。
- **ERNIE**（百度，2019）：BERT 的改进，mask 的不只是字，还盖住词、实体、短语等知识单元，中文任务表现突出。
- **Big Bird**（Google，2020）：用稀疏注意力（局部窗口 + 随机 + 全局 token），把 BERT 的 O(n²) 计算量降下来，支持超长文本。
- 演进脉络：ELMo（上下文词向量）→ BERT（双向预训练）→ GPT（单向生成），参数量一路暴涨（94M → 175B）。

## 二、BERT 的预训练

- BERT 可以做 Transformer 的 encoder：输入多少向量就输出多少向量。

### 任务1：Masked Language Model（MLM，填空）

- randomly masking some tokens：随机盖住部分输入文字，用一个特殊 token 代替；或者随机找一个别的字替换掉被盖住的字。
- 训练时随机决定用哪种方法、随机决定处理哪个字。
- 把被盖住字的输出向量做 linear transform → softmax → 输出概率，希望预测出原来的字（这是在加强抗干扰能力？）。

### 任务2：Next Sentence Prediction（NSP）

- cls 代表开始，sep 代表两个句子之间的间隔。
- cls 经过线性变换（+softmax）判断这两个句子是不是相接的。
- ALBERT 的改进：SOP（Sentence Order Prediction，句子顺序预测）。

### 预训练结果

- BERT 学到了：怎么填空、知道被挡住的字是什么、判断两个句子是否连续。
- 之后就可以拿来做其他下游任务：fine-tune 微调（pre-train → fine-tune）。
- 判断任务集：GLUE。

## 三、怎么使用 BERT（下游任务微调）

- 在训练下游模型时需要配对好的训练资料。

### 场景1：分类问题（一句话 → 一个类别）

- 例：判断一句话的正负面。
- Linear 和 BERT 要一起训练：Linear 随机初始化，BERT 用预训练参数（已学会做填空）。
- Linear 输出经过 softmax 得到类别。

### 场景2：输入输出等长的序列标注

- 例：词性判断。
- 输入 cls + 句子；对后续每个 token 的向量分别做分类任务。

### 场景3：输入两个句子 → 输出一个类别

- 例：Natural Language Inference（NLI，自然语言推理）：判断两个句子的关系——蕴含（entailment，赞成）、矛盾（contradiction）、中立（neutral）。
- 只用看 cls 的输出；需要配对的训练资料。

### 场景4：抽取式问答（Extraction-based QA）

- 输入：一篇文章 + 一个问题；输出：两个数字。
- 两个数字分别表示答案在文章里的开始位置和结束位置，直接从文章里提取出答案。
- 输入格式：cls question sep document。
- 需要从头训练的是两个向量：
  - 一个和所有文章 token 的向量做 inner product → softmax，输出答案开始位置；
  - 另一个（蓝色）向量做同样的事情，预测答案结束位置。
- 理论上的输入没有限制，实际上有限制——不然运算会非常复杂。

## 四、BERT 相关研究

- BERT 的训练非常困难，后续直接拿过来用就很方便——他直接就会填空题。
- **BERT Embryology（胚胎学）**：研究在训练过程中填空能力是怎么一步步增强的。
- **pre-train 一个 seq2seq model**：对 encoder 的输入进行扰动，经过 cross attention，由 decoder 恢复被弄坏前的输入。

## 五、Why does BERT work?

- embedding 是向量化？投影到向量空间？
- BERT 可以考虑上下文，让每个向量处在向量空间中的各自位置，达到分辨语义的目的——看起来也就有了真正读懂句子的能力。
- 反例思考：假如把在文字上训练的 BERT 用到 DNA 预测——把基本单位对应到特定的文字，就可以把 DNA 的类别用一串文字来表示，然后输出类别。
- 但即使是很随便的乱码，BERT 也能处理得很好——那 BERT 到底是不是真的能看懂语义？（存疑）

## 六、多语言 BERT

- 如果有预训练（学会多种语言的填空），那么只训练其中一种语言的特定任务，其他语言也会学会。
- 解释：也许不同语言对机器来说没什么差别——在向量空间的位置是跳跃的（共享语义空间）。
- 前提：需要有大量的训练资料。
- 但不同语言的符号之间的差别 BERT 是能认出来的——它不会在做英文填空的时候输出中文。
- 如果输出的 embedding 加上中英文之间的平均向量差，再经过 reconstruction，就可以重新输出为中文。
- 结论：多语言 BERT 里蕴含不同语言的信息差，并且可以被固定地发现与磨平。

## 七、GPT 系列

- 任务：预测接下来出现的会是什么（语言模型）。
- linear transform → softmax → cross-entropy。
- 像 decoder：只能看左边的（之前输入的）token，单向。
- Few-shot Learning：没有梯度下降，即 In-context Learning（靠上下文示例）。
- One-shot Learning：只给一个示例。
- Zero-shot Learning：不给示例。
