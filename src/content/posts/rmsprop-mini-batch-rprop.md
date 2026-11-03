---
title: "Optimizer"
date: "2026-11-02"
description: "SGD, Adagrad, RProp, RMSProp, Adam, AdamW"
author: "Soleun Won"
tags:
  - "Optimization"
  - "Deep Learning"
---

## 1. RProp

### - 문제 인식: 모든 weight에 적절한 하나의 Global Learning Rate 설정의 어려움

같은 learning rate를 쓰더라도 weight마다 gradient의 크기가 크게 차이난다면, 어떤 weight는 너무 크게 움직이고 어떤 weight는 거의 움직이지 않을 수 있다. 

**해결 방법 제시: gradient의 sign만 반영**

$$ 
\frac{g}{\lvert g \rvert} = \text{sign}(g)
$$

- **장점**: gradient가 아주 작더라도 0이 아니기만 하면 일정한 크기로 업데이트할 수 있다. 즉, gradient가 거의 0에 가까운 평평한 구간(plateau)에서도 학습이 멈추지 않는다.
- **한계**: 실제로는 weight마다 적절한 업데이트 크기가 다를 수 있음에도 모든 weight가 같은 크기(1)로 업데이트된다. 

따라서 RProp은 **gradient의 sign만 사용하는 아이디어**와 **weight마다 step size를 따로 조절하는 아이디어**를 결합한다.

---
## 2. RProp은 step size를 어떻게 조절할까?

step size = 해당 weight를 한 번 업데이트할 때 이동하는 크기 <br>
RProp은 **이전 gradient와 현재 gradient의 부호를 비교**해서 step size 조절한다.

**Case 1. 이전 gradient 부호 = 현재 gradient 부호**

weight를 계속 같은 방향으로 움직이라는 의미로, 해당 weight의 step size를 증가시킨다. (예: 1.2배)

<br>

**Case 2. 이전 gradient 부호 ≠ 현재 gradient 부호**

Optimal point를 향해 이동하다가 너무 크게 움직여서 Optimal point를 지나쳐 반대편으로 넘어갔을 가능성이 있다는 의미로 해석할 수 있다. 해당 weight의 step size를 감소시킨다. (예: 0.5배)

실제 구현에서는 step size가 무한히 커지거나 지나치게 작아지는 것을 막기 위해 하한과 상한을 설정한다. 이는 Mike Shuster의 조언으로, 대략 0.000001 이상 50 이하의 범위를 사용한다.

<br>

**RProp 정리**: gradient의 부호로 업데이트 방향을 정하고, 부호가 유지되는지 바뀌는지에 따라 weight별 step size를 조절한다.

---
## 3. 왜 RProp은 mini-batch에서 잘 안 될까?

**예시: 어떤 weight의 gradient가 10번의 mini-batch에서 다음과 같이 계산되었다고 가정하자.**

- gradinet: +0.1 (9번)
- gradinet: -0.9 (1번)

Mini-batch SGD의 weight 업데이트 수식은 다음과 같다.

$$
w_{t+1}=w_t-\eta g_t
$$

<br>

$w_t$: 현재 weight

$\eta$: learning rate

$g_t$: 현재 mini-batch에서 계산한 gradient

<br>
주어진 10개의 gradient를 적용해 전체 weight 변화량을 구하면 아래와 같다.

$$
\begin{aligned}
\Delta w
&=-\eta\sum_{t=1}^{10}g_t\
&=-\eta[9(0.1)+(-0.9)]\
&=0
\end{aligned}
$$

따라서 주어진 gradient가 유지된다고 가정하면, 10번의 mini-batch 업데이트 이후 weight는 초기값과 동일하게 유지된다.
SGD는 연속된 mini-batch의 gradient를 크기에 따라 가중하여 반영하므로, 여러 gradient의 평균을 내는 것과 유사한 효과를 갖는다.

반면 RProp은 gradient의 크기를 무시하고 부호만 반영한다. 즉, 순방향으로 9번, 역방향으로 1번 업데이트하는 것으로 계산된다. 이때 gradient가 +0.1이든 -0.9이든 동일한 step size가 적용되므로, 크기가 큰 -0.9의 영향력이 +0.1 하나와 동일하게 취급된다. 따라서 전체적으로 한쪽 방향으로 weight가 이동하게 된다.

따라서 mini-batch 환경에서도 gradient의 크기 변화를 고려하면서 업데이트 크기를 적절하게 조절할 방법이 필요하다. 

<br>

**정리**: rprop은 full-batch 학습에 적합한 방법이다.

---
## 4. RMSProp: RProp의 mini-batch 버전

**인접한 mini-batch끼리는 나누는 값이 비슷하도록 강제하면 어떨까?**

RMSProp은 RProp의 한계를 해결하기 위해 제안된 방법으로, **gradient 제곱의 지수이동평균(Exponential Moving Average, EMA)**을 활용한다.

$$ 
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{\text{MeanSquare}(w,t)} + \epsilon} 
$$

### - EMA

$$
\text{MeanSquare}(w,t) = 0.9 \times \text{MeanSquare}(w,t-1) + 0.1 \times g_t^2
$$

<br>

$\text{MeanSquare}(w,t-1)$: 이전 시점까지 계산된 gradient 제곱의 이동평균

$g_t^2$: 현재 mini-batch에서 계산된 gradient의 제곱

$0.9$: 과거 정보에 부여하는 가중치

$0.1$: 현재 gradient에 부여하는 가중치

<br>

즉, MeanSquare(w,t)은 과거 정보 90%와 현재 gradient 제곱의 10%를 반영하여 계산한다.
여기서 중요한 점은 gradient 자체가 아니라 gradient의 제곱을 사용한다는 것이다. 따라서 양수와 음수의 방향은 고려하지 않고, gradient의 크기 정보만 추적할 수 있다.

<br>

**예시: decay rate = 0.9 이라 한다면**

$$
\begin{aligned}
MeanSquare_t
=&\ 0.1g_t^2\
&+0.09g_{t-1}^2\
&+0.081g_{t-2}^2+\cdots
\end{aligned}
$$

따라서 최근 gradient의 크기에 더 큰 가중치를 부여하고, 오래된 gradient의 영향은 점점 줄어들게 된다.

<br>

다음으로, 계산한 EMA를 이용하여 weight를 업데이트한다.

$$ 
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{\text{MeanSquare}(w,t)} + \epsilon} 
$$

<br>

$w_t$: 현재 weight

$\eta$: Learning Rate

$g_t$: 현재 mini-batch의 gradient

$\text{MeanSquare}(w,t)$: gradient 제곱의 지수이동평균

$\epsilon$: 0으로 나누는 것을 방지하기 위한 작은 값

<br>

여기서 EMA의 제곱근으로 현재 gradient를 나누는 이유는 gradient의 크기를 정규화하기 위함이다.
EMA를 이용하면 최근 gradient의 크기에 따라 업데이트 크기를 자동으로 조절할 수 있다.

- 최근 gradient의 크기가 커지면, 분모가 커지면서 weight의 업데이트 폭이 줄어든다.
- 반대로 최근 gradient의 크기가 작아지면, 분모가 작아지면서 상대적으로 더 큰 업데이트가 가능해진다.

즉, RMSProp은 **최근 gradient의 크기를 기준으로 각 weight의 step size(업데이트 크기)를 적응적으로 조절하는 알고리즘**이다.

---
## 5. Adagrad와 비교

### - Adagrad
$$ 
G_t = G_{t-1} + \left(\frac{\partial L}{\partial w_t}\right)^2 
$$

<br>

$G_t$: 현재까지 누적된 gradient 제곱의 합

$G_{t-1}$: 이전 시점까지 누적된 gradient 제곱의 합

$g_t$: 현재 시점의 gradient

<br>

$$
w_{t+1}
=
w_t -
\frac{\eta}{\sqrt{G_t}+\epsilon}
\frac{\partial L}{\partial w_t}
$$

<br>

### - RMSProp
$$
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{\text{MeanSquare}(w,t)} + \epsilon} 
$$

RMSProp과의 수식을 비교하면 둘의 차이를 더 쉽게 이해할 수 있다.

|  | Adagrad | RMSProp |
|---|---|---|
| 과거 gradient 제곱 | 전부 **동일한 비중으로 누적** | **EMA** 로 과거에 점점 작은 가중치 부여 |


### - Adagrad의 문제점

gradient 제곱을 계속 더하기 때문에 누적합은 절대 줄어들지 않는다. 즉 분모가 계속 커지므로 학습이 진행될수록 실제 step size(learning rate를 분모로 나눈 값)가 점점 작아지고, 심해지면 학습이 사실상 멈추게 된다.

RMSProp은 EMA를 사용해 과거 정보의 영향을 점점 줄이기 때문에 이 문제를 피할 수 있다.

<br>

## 정리

- **RProp**: gradient의 부호만 사용하고, weight마다 step size를 조절. full-batch 학습에 적합.
- **RMSProp**: gradient 제곱의 지수 이동 평균으로 나누어, rprop의 아이디어를 mini-batch에서도 쓸 수 있게 만든 방법.
