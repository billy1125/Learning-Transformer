# Transformer 與深度學習經典論文推薦

以下論文依照「理解深度」與「學習路徑」排序，重點是幫助建立扎實的模型理解能力。

---

# 一、核心必讀（Transformer 主幹）

## 1. Attention Is All You Need (2017)
- Vaswani et al.

**為什麼讀**
- Transformer 的原點
- attention 機制的完整定義

**你要抓的重點**
- scaled dot-product attention
- multi-head 的動機
- positional encoding 的必要性

---

## 2. BERT: Pre-training of Deep Bidirectional Transformers (2018)
- Devlin et al.

**為什麼讀**
- Transformer 如何變成「語言理解模型」

**重點**
- pretrain / finetune paradigm
- masked language modeling

---

## 3. GPT: Improving Language Understanding by Generative Pre-Training (2018)
- Radford et al.

**為什麼讀**
- 自回歸（autoregressive）路線

**重點**
- causal masking
- language modeling 作為通用任務

---

# 二、理解 Attention 本質

## 4. The Annotated Transformer（強烈推薦）
- Harvard NLP

**為什麼讀**
- 最清楚的實作級解釋
- 幾乎是標準教材

---

## 5. On the Relationship Between Self-Attention and Convolution (2019)
- Cordonnier et al.

**為什麼讀**
- 幫助理解 attention 的本質

**重點**
- attention 可以模擬 convolution

---

# 三、表示能力與理論基礎

## 6. Universal Approximation Theorem（1989）
- Hornik

**為什麼讀**
- 神經網路的理論基礎

**重點**
- NN 是函數逼近器
- 可逼近 ≠ 容易訓練

---

## 7. Deep Residual Learning for Image Recognition (ResNet, 2016)
- He et al.

**為什麼讀**
- 解決深層網路訓練困難

**重點**
- residual connection 的作用
- optimization 改善

---

# 四、訓練與優化

## 8. Adam: A Method for Stochastic Optimization (2015)
- Kingma & Ba

**為什麼讀**
- 最常用的 optimizer

**重點**
- adaptive learning rate
- momentum + scaling

---

## 9. Understanding the Difficulty of Training Deep Feedforward Neural Networks (2010)
- Glorot & Bengio

**為什麼讀**
- 初始化的重要性

**重點**
- 梯度消失 / 爆炸
- Xavier initialization

---

# 五、進階延伸（仍屬可讀）

## 10. An Image is Worth 16x16 Words: Vision Transformer (ViT, 2020)
- Dosovitskiy et al.

**為什麼讀**
- Transformer 延伸到 vision

**重點**
- patch embedding
- attention 作為通用架構

---

# 六、建議閱讀順序

1. Attention Is All You Need  
2. The Annotated Transformer  
3. BERT 或 GPT（先選一個）  
4. ResNet  
5. Adam  
6. Xavier Initialization  
7. Self-Attention vs Convolution  
8. Vision Transformer  
9. Universal Approximation Theorem（可穿插）

---

# 七、學習目標（比讀完更重要）

完成後你應該能回答：

- attention 的本質是什麼？
- Transformer 為什麼優於 RNN？
- residual connection 在解決什麼問題？
- 神經網路在數學上做的是什麼？

---

# 八、總結

- 這份清單足以建立扎實基礎
- 重點在理解，不在數量
- 建議反覆閱讀關鍵論文（尤其 Transformer）
