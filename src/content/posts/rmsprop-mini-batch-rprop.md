---
title: "RMSProp"
date: "2026-11-02"
description: "rprop의 아이디어와 mini-batch에서 잘 동작하지 않는 이유를 짚어 보고, 이를 보완한 RMSProp과 Adagrad와의 차이를 정리"
author: "Soleun Won"
tags:
  - "Optimization"
  - "RMSProp"
  - "Deep Learning"
---

## 1. rprop

### 문제 제시: 모든 weight에 적절한 하나의 Global Learning Rate 설정의 어려움

같은 learning rate를 쓰더라도 weight마다 gradient의 크기가 크게 다르면, 어떤 weight는 너무 크게 움직이고 어떤 weight는 거의 움직이지 않을 수 있기 때문이다.

### 해결 방법: gradient의 sign만 반영

$$ \frac{g}{\lvert g \rvert} = \text{sign}(g) $$

- **장점**: gradient가 아주 작더라도 0이 아니기만 하면 일정한 크기로 업데이트할 수 있다. gradient가 거의 0에 가까운 평평한 구간(plateau)에서도 학습이 멈추지 않는다.
- **한계**: 모든 weight가 같은 크기로 업데이트된다. 실제로는 weight마다 적절한 업데이트 크기가 다를 수 있다.

그래서 rprop은 **gradient의 부호만 사용하는 아이디어**와 **weight마다 step size를 따로 조절하는 아이디어**를 결합한다.

## 2. rprop은 step size를 어떻게 조절할까?

step size = 해당 weight를 한 번 업데이트할 때 이동하는 크기. 
rprop은 **이전 gradient와 현재 gradient의 부호를 비교**해서 step size 조절한다.

**Case 1. 이전 gradient 부호 = 현재 gradient 부호**

weight를 계속 같은 방향으로 움직이라는 의미. 이 weight의 step size를 증가 (예: 1.2배)

**Case 2. 이전 gradient 부호 ≠ 현재 gradient 부호**

최적점을 향해 이동하다가 너무 크게 움직여서 최적점을 지나쳐 반대편으로 넘어갔을 가능성이 있다는 의미. 이 weight의 step size를 감소 (예: 0.5배)

실제 구현에서는 step size가 무한히 커지거나 지나치게 작아지는 것을 막기 위해 하한과 상한을 설정한다. 이는 Mike Shuster의 조언으로, 대략 0.000001 이상 50 이하의 범위를 사용한다.

**정리**: gradient의 부호로 업데이트 방향을 정하고, 부호가 유지되는지 바뀌는지에 따라 weight별 step size를 조절한다.

## 3. 왜 rprop은 mini-batch에서 잘 안 될까?

mini-batch SGD는 learning rate가 작을 때, 연속된 mini-batch들의 gradient를 **평균**내는 효과가 있다. 

예시) 어떤 weight의 gradient가 10번의 mini-batch 아래와 같다.

- 9번은 +0.1
- 1번은 -0.9

실제로 mini-batch SGD는 gradient의 크기까지 반영해서 합산하므로, 이 값들의 합이 0이 되어 weight가 거의 움직이지 않는다.

하지만 rprop은 크기를 무시하고 부호만 반영한다.  즉 순방향 9번, 역방향 1번으로 계산되어 weight가 한쪽 방향으로 계속 쏠리게 된다. 크기가 큰 -0.9의 영향이 +0.1 하나와 똑같이 취급되기 때문이다.

**정리**: rprop은 full-batch 학습에 적합한 방법

## 4. RMSProp: rprop의 mini-batch 버전

**인접한 mini-batch끼리는 나누는 값이 비슷하도록 강제하면 어떨까?**

RMSProp은 **gradient 제곱의 이동 평균**으로 나눈다. 방향이 아니라 크기 정보를 활용하는 것.

- 새로운 값 = 과거 평균의 0.9 + 현재 gradient 제곱의 0.1
- 0.9라는 비율 덕분에 시간이 지날수록 과거의 영향은 점점 작아짐.

$$ w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{\text{MeanSquare}(w,t)} + \epsilon} $$

## 5. Adagrad와 비교

$$ G_t = G_{t-1} + \left(\frac{\partial L}{\partial w_t}\right)^2 $$

|  | Adagrad | RMSProp |
|---|---|---|
| 과거 gradient 제곱 | 전부 **동일한 비중으로 누적** | **EMA** 로 과거에 점점 작은 가중치 부여 |


### Adagrad의 문제점

gradient 제곱을 계속 더하기 때문에 누적합은 절대 줄어들지 않는다. 즉 분모가 계속 커지므로 학습이 진행될수록 실제 step size(learning rate를 분모로 나눈 값)가 점점 작아지고, 심해지면 학습이 사실상 멈추게 된다.

RMSProp은 EMA를 사용해 과거 정보의 영향을 점점 줄이기 때문에 이 문제를 피할 수 있다.

## 정리

- **rprop**: gradient의 부호만 사용하고, weight마다 step size를 조절. full-batch 학습에 적합.
- **RMSProp**: gradient 제곱의 지수 이동 평균으로 나누어, rprop의 아이디어를 mini-batch에서도 쓸 수 있게 만든 방법.
