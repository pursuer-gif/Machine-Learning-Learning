---
tags: [concept, deep-learning, batch-norm, transformer, seq2seq, lee-hung-yi]
aliases: ["ML Course Notes 3", "机器学习听课笔记3"]
sources: [机器学习/来源/李宏毅机器学习深度学习2026版来源.md]
created: 2026-07-08
updated: 2026-07-08
---

# 机器学习听课笔记 3：Batch Normalization 与 Transformer

> 李宏毅课程

---

## 一、Batch Normalization

对整个数据集做标准化计算量太大，一般在 batch 上做，但 batch 要足够大。

**作用**：改变 loss function 的地形。多个变量对 loss 的影响差别较大时，不同方向的斜率、坡度差值较大——归一化后各方向尺度一致，训练更稳。

### 标准化流程

Feature normalization（标准化），输出也需要 normalization。在激活函数之前或之后做差别不大，根据不同的函数区分挑选。

做完标准化后还要对每一个输出值进行线性处理：

$$\hat{z}_i = \gamma \cdot \frac{z_i - \mu}{\sigma + \epsilon} + \beta$$

- $\mu$、$\sigma$：当前 batch 的均值和标准差
- $\gamma$、$\beta$：可学习的缩放和偏移——模型可以自己学"归一化到多少"，不想归一化时通过学习恢复原始分布
- $\epsilon$：防止除零

不同参数经过标准化后产生关联，可以看成一个巨大的 network。

在实际生产中，可能达不到设定的 batch 标准就要进行 batch normalization。

---

## 二、Transformer

### Seq2Seq 应用场景

| 应用 | 说明 |
|------|------|
| 语音辨识 | Speech Recognition |
| 语音转换 | Voice Conversion |
| 机器翻译 | Machine Translation |
| 语音翻译 | Speech Translation |
| 摘要 | Summarization |
| 情感分析 | 判断评价的正负面 |
| 语法剖析 | Parsing |
| Multi-label Classification | 一个东西对应多个分类 |
| Object Detection | 甚至也可以用 seq2seq 硬做 |

大多数自然语言处理可以看成 QA 问题。

### Encoder → Decoder 架构

Encoder 输入一排向量，输出一排向量。每层 Block 内部四步：

**第 1 步** — Self-Attention：输出 = 输入 + 自注意力结果（残差连接）

**第 2 步** — Layer Normalization

**第 3 步** — Feed-Forward Network

**第 4 步** — 残差连接 → Layer Normalization → 输出

重复 N 次 Block。还要加入 positional encoding。

### 各步骤速查

| 步骤 | 干什么 | 为什么 |
|------|--------|--------|
| Embedding | token → 固定维向量 | 离散符号 → 可计算 |
| Positional Encoding | 加上位置标记 | Self-Attention 不感知顺序 |
| 多头自注意力 | 每个位置看向整个序列 | 捕捉上下文，多头关注不同关联 |
| 掩码自注意力 | 同上但只看自己及左边 | 生成时不能偷看未来 |
| 残差连接 | 输出 = 输入 + 子层结果 | 梯度高速通道，防梯度消失 |
| LayerNorm | 归一化 | 稳定深层训练数值分布 |
| FFN | 每位置独立全连接 | SA 管"跟谁交互"，FFN 管"自己想清楚" |
| Cross-Attention | Decoder Q → Encoder K,V | 生成时找最相关原始信息 |
| Linear + Softmax | 映射词表转概率 | 选最可能 token |

### 结构图

### Q、K、V 详解

Self-Attention 的核心运算。每个输入向量 $\mathbf{a}_i$ 通过三个不同的权重矩阵产生三个向量：

| 名称 | 全称 | 公式 | 含义 |
|------|------|------|------|
| **Q** | Query | $\mathbf{q}_i = W_q \mathbf{a}_i$ | "我在找什么"——当前词想要关注什么样的信息 |
| **K** | Key | $\mathbf{k}_i = W_k \mathbf{a}_i$ | "我是什么"——当前词能提供什么样的信息 |
| **V** | Value | $\mathbf{v}_i = W_v \mathbf{a}_i$ | "我携带什么"——当前词实际携带的信息内容 |

**计算流程：**

1. **算注意力分数**：每个词的 Q 去跟所有词的 K 做点积，得到"这个词跟那个词有多相关"

   $$\text{score}_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$$

2. **Softmax 归一化**：把所有分数变成 [0,1] 的权重，和为 1

   $$\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_k \exp(\text{score}_{ik})}$$

3. **加权取信息**：用归一化后的权重去乘每个词的 V，求和

   $$\mathbf{b}_i = \sum_j \alpha_{ij} \mathbf{v}_j$$

**一句话总结**：Q 问"谁跟我有关"，K 答"我可以是这样"，匹配上了就用 V 把那个词的信息提取过来。三个矩阵 $W_q$、$W_k$、$W_v$ 是**学出来的**，不是手工设计的。

**多头注意力**：多套并行的 $W_q$、$W_k$、$W_v$，各头各自学不同的关注偏好：

```
输入 a
  │
  ├── 头1: Wq₁,Wk₁,Wv₁ → 可能学出"语法搭配"
  ├── 头2: Wq₂,Wk₂,Wv₂ → 可能学出"语义相似"
  └── 头3: Wq₃,Wk₃,Wv₃ → 可能学出"指代关系"
```

不同 W 矩阵 = 不同的观察视角。不是 Q 驱动 K，而是**每套 W 自己决定这个头关心什么类型的关联**。各头的输出拼接后过一个 $W_o$ 投影回原维度。

**V 也是并行的**——每个头有自己的 $W_v$，同一个输入 token 在不同头里提取的信息不同：头 1 的 V 取词义，头 2 的 V 取词性信息，头 3 的 V 取实体指向。

**Cross-Attention 中的 QKV**：Decoder 的交叉注意力层，Q 来自 Decoder 自己的上一层的输出（"我要找什么"），K 和 V 来自 Encoder 的最终输出（"原始输入里有什么"）。相当于生成每个词时，都去原始句子中找最相关的信息。原始论文中不管 Encoder 多少层，Decoder Cross-Attention 吃的是 Encoder 最终输出（可有变种）。

**Cross-Attention 流程**：Decoder 先吃 BEGIN，输出乘 $W_q$ 得 Q，与 Encoder K 做点积，再用 Encoder 各层 V 加权求和，经全连接得到最终结果。

---

## 四、Decoder 生成过程

以中文为例，输出约三千常用字。每一步 Softmax 取概率最大的字输出，再当作下一次输入，反复持续。

### Masked Self-Attention

生成第 t 个字时只看 1~t，不能偷看未来。算完注意力分数后、Softmax 前，把未来位置分数设 −∞ → 权重自动归零。

### 开始与停止

从 **BEGIN**（one-hot）开始，用 **END** 通知停止。两者可用同一符号。

### Teacher Forcing 与 Exposure Bias

训练时喂正确答案（Teacher Forcing），避免一步错步步错。但带来 **Exposure Bias**——训练时看到的是正确输入，推理时看到自己可能错误的输出，分布不一致。缓解：训练时偶尔给 decoder 喂带噪声的输入。训练目标：每个位置的 Cross-Entropy 最小。

---

## 五、AT vs NAT

| | AT（Autoregressive） | NAT（Non-Autoregressive） |
|---|---|---|
| 生成 | 逐字串行，Decoder 重复 N 次 | 一次生成整句，并行一次 |
| 速度 | 慢 | 快 |
| 长度 | END 自然停止 | 需额外模型预测长度或超长 BEGIN 再截断 |
| 表现 | 更好 | 较差（Multi-modality：同一输入多种合理输出，NAT 只给一种） |

---

## 六、高级技巧

### Copy Mechanism

输出可直接从输入搬——人名、专有名词、编号。Chat-bot、摘要常用。

### Guided Attention

约束 Attention 遵循特定模式。如语音合成时 Attention 应从左到右扫。

### Beam Search

Greedy Decoding 每步选最大概率 → 局部最优 ≠ 全局最优。穷举不现实 → Beam Search 每步保留 top-K 候选路径。注意：需要多样性的任务（如对话）中不一定好，最高分不总是人类认为最好的。

### BLEU Score

评估翻译/生成质量的指标。训练用 Cross-Entropy（可微），评估看 BLEU（不可微，但与人类判断更一致）。**训练和评估指标可以不同。**

---

| 步骤 | 干什么 | 为什么 |
|------|--------|--------|
| Input / Output Embedding | 把 token 变成固定维度的向量 | 离散符号 → 可计算的连续向量空间 |
| Positional Encoding | 给每个向量加上位置标记 | Self-Attention 本身不感知顺序 |
| Multi-Head Self-Attention | 每个位置看向整个序列，加权聚合 | 捕捉上下文依赖，多头关注不同关联 |
| Masked Self-Attention（Decoder） | 同上，但只能看到自己及左边 | 生成时不允许偷看未来 |
| 残差连接 | 该层输出 = 输入 + 子层处理结果 | 给梯度留高速通道，防梯度消失 |
| Layer Normalization | 把加完残差的向量归一化 | 稳定每层数值分布 |
| Feed Forward Network | 对每个位置独立做两层全连接 | SA 负责"跟谁交互"，FFN 负责"想清楚" |
| Cross-Attention（Decoder） | Q 来自 Decoder，K/V 来自 Encoder | 生成时去 Encoder 找最相关原始信息 |
| Linear + Softmax | 映射到词表大小，转概率 | 选当前最可能输出的 token |

---

Related: [[机器学习/概念/自注意力与Transformer|自注意力与Transformer]], [[机器学习/概念/神经网络训练排查|神经网络训练排查]], [[机器学习/概念/强化学习概览|强化学习概览]], [[机器学习听课笔记2|机器学习听课笔记 2]]
