---
title: "Show and Tell: A Neural Image Caption Generator"
date: "2026-11-17"
description: "논문 정리"
author: "Soleun Won"
tags:
  - "CNN"
  - "RNN"
  - "LSTM"
---

## Show and Tell: A Neural Image Caption Generator


## Abstract

이 논문은 이미지를 CNN으로 벡터 하나로 바꾸고, 그 벡터를 받은 LSTM이 문장을 한 단어씩 생성하는 모델 NIC(Neural Image Caption)를 제안한다. 모델 전체를 하나의 신경망으로 묶어 처음부터 끝까지 함께 학습한다는 점이 핵심이다.

<br>
---

## 1. Introduction

1장에서는 Machine translation의 encoder-decoder 구조에서 encoder RNN을 CNN으로 바꾸면 이미지를 문장으로 번역할 수 있다는 아이디어를 제시한다. 


### - 기존 연구와 본 논문의 차이점

이전 연구는 대부분 객체 검출, 문장 생성 같은 하위 문제의 기존 해법을 이어 붙이는 방식이었다. 본 논문은 하나의 joint 모델을 제안한다. 이미지 I를 입력받아, 이미지를 적절히 설명하는 단어열 S = {S1, S2, …}가 나올 확률 p(S|I)를 최대화하도록 학습한다. 각 단어 St는 주어진 dictionary에서 나온다.


### - Machine translation에서 가져온 아이디어

Machine translation은 원문 S를 번역문 T로 바꾸는 일로, p(T|S)를 최대화하는 문제로 볼 수 있다. 최근 연구는 RNN만으로 훨씬 단순하게 번역하면서도 최고 성능에 도달했다.

그 방식은,

1. Encoder RNN이 원문을 읽어 고정 길이 벡터 표현으로 바꾼다.
2. 이 벡터가 decoder RNN의 초기 hidden state가 된다.
3. Decoder RNN이 번역문을 생성한다.


### - NIC

Machine translation 구조에서 encoder RNN을 deep CNN으로 바꾼다. 이것이 본 논문에서 주장하는 NIC.

<br>

---

## 3. Model

3장은 NIC를 이미지가 주어졌을 때 정답 문장의 확률을 최대화하는 확률 모델로 정의하고, 그 확률을 LSTM으로 계산하는 방법을 설명한다. 


### - 확률 모델

Machine translation은 강력한 시퀀스 모델로 정답 번역의 확률을 직접 최대화해 최고 성능을 냈다. 논문 저자들은 입력 문장 대신 이미지를 넣어 같은 원리를 적용한다. 아래 식은 (1)

$$
\theta^{*} = \arg\max_{\theta} \sum_{(I,S)} \log p(S \mid I; \theta)
$$

- θ는 모델 파라미터, I는 이미지, S는 그 이미지의 정답 문장
- 모든 (이미지, 문장) 쌍에 대해 로그 확률의 합이 가장 커지는 θ를 찾는다.

문장 S는 길이가 정해져 있지 않다. 그래서 chain rule로 단어 하나씩의 확률로 분해한다. 아래 식은 (2)

$$
\log p(S \mid I) = \sum_{t=0}^{N} \log p(S_t \mid I, S_0, \ldots, S_{t-1})
$$

- N은 해당 문장의 길이
- 학습 때는 이 로그 확률의 합을 전체 학습 데이터에 대해 stochastic gradient descent로 최적화한다.

<br>

### - RNN으로 확률을 모델링하기

식 (2)의 각 항은 이미지와 앞선 단어들이 주어졌을 때 다음 단어의 확률이다. 조건으로 들어가는 단어 수가 t마다 다르므로, 이를 고정 길이의 hidden state(메모리) h\_t로 표현한다. 새로운 입력 x\_t이 들어오면 non-linear 함수 f로 h\_t를 업데이트한다. 아래 식은 (3)

$$
h_{t+1} = f(h_t, x_t)
$$

이 RNN을 구체화하려면 두 가지를 정해야 한다.

1. 정확히 f가 뭔지
2. image랑 word가 어떻게 x\_t로 주어지는지

| 설계 선택 | 논문의 결정 | 이유 |
| --- | --- | --- |
| f의 형태 | LSTM | 번역 같은 시퀀스 과제에서 최고 성능을 보였다 |
| 이미지 표현 | CNN | 객체 인식·검출에서 최고 성능이다. Batch normalization을 쓰는 모델 \[12\]을 택했고, 이 모델은 ILSVRC 2014 분류 대회 기준 최고 성능이다. Transfer learning으로 다른 과제에도 일반화된다 |
| 단어 표현 | Embedding model |  |

<br>

### 3.1 LSTM-based Sentence Generator

f를 고를 때의 기준은 vanishing and exploding gradient를 다룰 수 있느냐이다. LSTM은 이 문제를 해결하려고 만들어진 RNN이다.

LSTM의 핵심은 **memory cell c** 이다. c는 매 시점, 지금까지 어떤 입력을 봤는지에 대한 정보를 담는다. Cell의 동작은 gate가 제어한다. Gate는 값에 곱해지는 layer이며, gate가 1이면 값을 유지하고 0이면 값을 0으로 만든다.

| Gate | 기호 | 역할 |
| --- | --- | --- |
| Forget gate | f | 현재 cell 값을 잊을지 결정 |
| Input gate | i | 입력을 읽어 들일지 결정 |
| Output gate | o | 새 cell 값을 출력할지 결정 |

Gate, cell 갱신, 출력의 정의는 식 (4)\~(9)다.

$$
\begin{aligned}
i_t &= \sigma(W_{ix} x_t + W_{im} m_{t-1}) \\
f_t &= \sigma(W_{fx} x_t + W_{fm} m_{t-1}) \\
o_t &= \sigma(W_{ox} x_t + W_{om} m_{t-1}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot h(W_{cx} x_t + W_{cm} m_{t-1}) \\
m_t &= o_t \odot c_t \\
p_{t+1} &= \mathrm{Softmax}(m_t)
\end{aligned}
$$

식을 읽는 순서는 다음과 같다.

1. **세 gate 계산 (식 4\~6)**: 현재 입력 x\_t와 직전 출력 m\_{t-1}에 각각 가중치를 곱해 더하고 sigmoid σ를 적용한다. 세 식은 형태가 같고 가중치 행렬만 다르다.
2. **Cell 갱신 (식 7)**: 이전 cell 값 c\_{t-1}에 forget gate를 곱해 남길 만큼 남긴다. 새로 들어온 정보 h(·)에는 input gate를 곱해 더한다.
3. **출력 (식 8)**: 새 cell 값에 output gate를 곱한 것이 LSTM의 출력 m\_t다.
4. **단어 예측 (식 9)**: m\_t를 Softmax에 넣어 모든 단어에 대한 확률분포 p를 얻는다.

<br>

**표기 해석**

- ⊙는 gate 값과의 곱이다.
- W matrix들은 모두 학습되는 파라미터다.
- σ는 sigmoid, h는 tanh, 여기에서 h는 식 (3)의 hidden state h\_t와 다른 기호
- LSTM의 출력을 m으로

논문에서는 이런 곱셈 형태의 gate가 exploding and vanishing gradient를 잘 다루기 때문에 LSTM을 안정적으로 학습할 수 있다고 설명한다.

<br>

### Training

LSTM은 이미지와 앞선 단어를 모두 본 뒤 문장의 각 단어를 예측하도록 학습된다. 

- 이미지와 문장의 단어마다 LSTM 복사본을 하나씩 둔다. 즉 같은 파라미터를 공유한다.
- 시점 t-1의 출력 m\_{t-1}이 시점 t의 LSTM으로 들어간다.
- 펼치면 recurrent 연결이 모두 feed-forward 연결로 바뀐다.

입력 이미지를 I, 정답 문장을 S = (S\_0, …, S\_N)이라 하면 펼친 절차는 식 (10)\~(12)다.

$$
\begin{aligned}
x_{-1} &= \mathrm{CNN}(I) \\
x_t &= W_e S_t, \quad t \in \{0 \ldots N-1\} \\
p_{t+1} &= \mathrm{LSTM}(x_t), \quad t \in \{0 \ldots N-1\}
\end{aligned}
$$

- 각 단어 S\_t는 사전 크기와 같은 차원의 one-hot 벡터다.
- S\_0는 특수한 시작 단어, S\_N은 특수한 종료 단어다. LSTM이 종료 단어를 내보내면 문장이 완성됐다는 신호다.
- 이미지와 단어는 같은 공간으로 매핑된다. 이미지는 비전 CNN으로, 단어는 word embedding W\_e로 매핑한다.

**이미지는 t = -1에 한 번만 입력된다.** 
: 경험에 의해 매 시점마다 이미지를 추가 입력으로 주면 결과가 더 나빠진다는 것을 실험으로 확인했다. 네트워크가 이미지의 노이즈를 이용하게 되어 더 쉽게 overfit하기 때문에 결과가 나빠지는 것이다.

Loss는 각 시점에서 정답 단어의 negative log likelihood를 더한 값이다. 아래 식은 (13)

$$
L(I, S) = -\sum_{t=1}^{N} \log p_t(S_t)
$$

이 loss를 LSTM의 모든 파라미터, 이미지 임베딩 CNN의 최상위 layer, word embedding W\_e에 대해 최소화한다.

<br>

### Inference

학습이 끝난 NIC로 이미지에서 문장을 생성하는 방법은 두 가지다.

1. Sampling

무작위로 뽑기

2. Beamsearch

b개 후보 두고 뽑기 (논문에서는 b=20 으로 둠)

