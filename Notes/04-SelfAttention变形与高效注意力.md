---
tags: [concept, self-attention, efficient-attention, transformer, lee-hung-yi]
aliases: ["ML Course Notes 4", "机器学习听课笔记4"]
sources: [机器学习/来源/李宏毅机器学习深度学习2026版来源.md]
created: 2026-07-08
updated: 2026-07-08
---

# 机器学习听课笔记 4：Self-Attention 的各种变形

> 李宏毅课程 · 高效注意力 / Attention 加速

---

## 一、动机：Self-Attention 的运算瓶颈

Self-Attention 是让 sequence 里的每个 token 都有不同的 Q 和 K，两两相乘组成 **Attention Matrix**（n×n）。

当 sequence 很长时，模块的主要运算集中在 self-attention 上。用来处理**图片或视频**（一张图 = 上万个 patch）时，Attention Matrix 的大小是平方级增长，运算量非常惊人。

因此出现了一大堆**简化/近似注意力**的方法，核心思路都是：**不把完整的 Attention Matrix 算出来**。

---

## 二、各种注意力变形

### 1. Local Attention（局部注意力）

部分注意力的值不用算出，只计算有需求的部分，不重要的直接用 0 代替——把信息集中在某个小范围。

> 类似 CNN 的局部感受野，可以加快运算，但不一定有好的结果。

### 2. Stride Attention（步长注意力）

跳跃着读取信息——有间隔地计算注意力矩阵，比如空两格、空三格。

```
计算:  ×  ✓  ×  ✓  ×  ✓
       位置 0  1  2  3  4  5
```

### 3. Global Attention（全局注意力）

在原始输入里加入**特殊的 token**（global token），它会从句子里每一个 token 收集信息，同时它的信息也会被所有 token 读到。

两种做法：
- 把输入里已有的某些 token 当成特殊 token
- 额外加几个特殊 token

**在 Attention Matrix 里的体现**：特殊 token 对应的**前 n 行和前 n 列**包含了所有需要的信息，其余部分归零不用算。

```
Attention Matrix (n×n)：
┌───────────────────────┐
│ ▓▓▓▓  ▓▓▓▓  ▓▓▓▓      │  ← 特殊 token 的行
│ ▓▓▓▓  ▓▓▓▓  ▓▓▓▓      │
│ ──────────────────────│
│ ▓▓▓▓                  │
│ ▓▓▓▓       0  0  0    │
│ ▓▓▓▓            0     │
└───────────────────────┘
   特殊 token 的列     其余位置=0，不用算
```

### 4. Random Attention（随机注意力）

可以混合不同的处理方式一起应用（比如一部分用 local，一部分用 global，一部分随机）。

### 5. 注意力值稀疏化（直接跳过低值）

考虑把注意力值很小的部分直接简化为 0、不计算——先**估计**哪些位置的值比较大，针对性地只算这些，小值的位置直接补 0。

---

## 三、Clustering（聚类法）

**核心思路**：把 token 分成若干类，只计算**落在同一个 Cluster** 里的 query 和 key 之间的注意力值，跨类的直接补 0。

关键在于**怎么分类**——有两种思路：

1. **人工分类**：凭借人类对问题的理解进行分类（human knowledge）
2. **机器自动学**：想办法让机器自行学习聚类方式

### Learnable Patterns（可学习的聚类模式）

通过另一个模型（NN）计算聚类归属——**jointly learned**（聚类和注意力一起被训练）。

具体做法（以 Sinkhorn Sorting Network 为例）：
- 用另一个神经网络计算出每个 token 属于哪个 cluster
- 得到的是解析度较低的软分配（soft assignment），然后做条件筛选/硬化
- 这样"怎么分组"这件事也是学出来的，实现全部机器自主学习

---

## 四、挑选代表性 Key（Representative Keys）

很多 Attention Matrix 是**低秩**的（秩比较低）——意味着信息有冗余，可以挑选出有代表性的 K 来。

| 方法 | 说明 |
|------|------|
| 直接挑 K | 挑出有代表性的 key，只算这些 key 的注意力 |
| 挑 Q 的问题 | 如果挑 query 而不是 key，输出维度会变少，可能达不到预期目的 |
| CNN 压缩 | 用 CNN 把长输入转化为短的，当做有代表性的 key |
| 线性变换筛选 | 乘上一个矩阵对 key 做线性变换，筛出重要的 |

> 注意：从 Loss 对 attention 值和 value vector 做梯度下降（gradient descent）——这里的"梯度下降"是听错/记混的概念，实际应为用梯度下降训练模型使低秩近似有效。

---

## 五、改变计算顺序：VK 先乘（V, K first）

可以交换输出最终注意力矩阵的计算顺序——**先把 V 和 K 的转置相乘**，再和 Q 组合，运算量会大幅减少。

**为什么能减少运算量**：矩阵乘法 (m×d) × (d×n) 的运算量是 m×d×n。

- 原始顺序：QK^T (n×n) → × V → 运算量 ~ n²d + n²d
- 新顺序：VK^T 先乘 → 运算量 ~ nd²（省掉了 n² 项）

**问题**：不经过 softmax 怎么得到结果？
**解决**：用一个特别的函数拆解证明——softmax 的归一化可以分解，让某些部分只用计算一次。这样 Q 还可以按自己的需要选取。

---

## 六、还需要 Q、K 吗？——Synthesizer 框架

一定要用 Q 和 K 来产生注意力值吗？

不一定——可以有不同的方法产生注意力值：
- 直接把注意力模式放进网络里让它**学习**（相当于有一个"万用"的注意力值）
- 让所有输入都有一个好的表现，不依赖 QK 匹配

这就是 **Synthesizer**（合成器）框架的思路：注意力权重不来自 QK 点积，而是直接生成/学习。

---

## 七、Summary

处理 sequence 不一定要用 attention，attention 也不一定要用 QK。这节课的变形路线：

```
human knowledge（人工设计）
  ├── clustering（聚类分组）
  ├── learnable pattern（可学习分组）
  ├── representative keys（挑代表性 K）
  ├── Q,K first → V,K first（交换计算顺序）
  └── Synthesizer（彻底不用 QK，直接学注意力）
```

---

Related: [[机器学习/概念/自注意力与Transformer|自注意力与Transformer]], [[机器学习听课笔记2|机器学习听课笔记 2]], [[机器学习听课笔记3|机器学习听课笔记 3]]
