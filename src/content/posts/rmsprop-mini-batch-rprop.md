---
title: "Optimizer"
date: "2026-11-02"
description: "SGD, Momentum, Adagrad, RProp, RMSProp, Adam, AdamW"
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

#### Case 1. 이전 gradient 부호 = 현재 gradient 부호

weight를 계속 같은 방향으로 움직이라는 의미로, 해당 weight의 step size를 증가시킨다. (예: 1.2배)

<br>

#### Case 2. 이전 gradient 부호 ≠ 현재 gradient 부호

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

**Why not force the number we divide by to be very similar for adjacent mini-batches?**

RMSProp은 RProp의 한계에서 고안된 방법으로, gradient 제곱의 지수이동평균(Exponential Moving Average, EMA)을 활용한다.

$$ 
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{\text{MeanSquare}(w,t)} + \epsilon} 
$$

### - EMA (Exponential Moving Average)

**Geoffrey Hinton 교수는 decay rate = 0.9로 두는 걸 제안했다.**

$$
\text{MeanSquare}(w,t) = 0.9 \times \text{MeanSquare}(w,t-1) + 0.1 \times g_t^2
$$

<br>

$\text{MeanSquare}(w,t-1)$: 이전 시점까지 계산된 gradient 제곱의 이동평균

$g_t^2$: 현재 mini-batch에서 계산된 gradient의 제곱

$0.9$: 과거 정보에 부여하는 가중치

$0.1$: 현재 gradient에 부여하는 가중치

<br>

$\text{MeanSquare}(w,t)$ 은 과거 정보 90%와 현재 gradient 제곱의 10%를 반영하여 계산한다.
여기서 중요한 점은 현재 gradient의 제곱을 사용한다는 것이다. 따라서 양수와 음수의 방향은 고려하지 않고, gradient의 크기 정보만 추적할 수 있다.

<br>

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
EMA를 이용하면 현재 gradient의 크기에 따라 업데이트 크기를 자동으로 조절할 수 있다.

- 현재 gradient의 크기가 커지면, 분모가 커지면서 weight의 업데이트 폭이 RMSProp 적용 전보다 줄어들게 된다.
- 반대로 현재 gradient의 크기가 작아지면, 분모가 작아지면서 상대적으로 더 큰 업데이트가 가능해진다.

RMSProp은 gradient가 가파른 곳에서는 step size를 줄여 조금씩 탐색하고, gradient가 평평한 곳에서는 step size를 크게 해 성큼성큼 탐색한다.
즉, RMSProp은 **최근 gradient의 크기를 기준으로 각 weight의 step size(업데이트 크기)를 적응적으로 조절하는 알고리즘**이다.

<br>

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
| 과거 gradient 제곱 | 전부 **동일한 비중으로 누적**(Sum) | **Decay rate**으로 가중평균(Averge) |

<br>

### - Adagrad의 문제점

gradient 제곱을 계속 더하기 때문에 누적합은 절대 줄어들지 않는다. 즉 분모가 계속 커지므로 학습이 진행될수록 실제 step size(learning rate를 분모로 나눈 값)가 점점 작아지고, 심해지면 학습이 사실상 멈추게 된다.

RMSProp은 EMA를 사용해 과거 정보의 영향을 점점 줄이기 때문에 분모가 줄어들 수도 있다. 분모가 결국 합으로 이뤄져 있어서 무한히 커진다고 생각할 수 있지만 가중치(0.9와 0.1)의 합이 정확히 1이기 때문에, 분모는 최근 기울기 제곱들의 평균적인 크기(Root Mean Square)에 수렴하게 된다.

**예시: gradient = 0 에 도달한 경우**

다음 EMA = (0.9 × 100) + (0.1 × 0) = 90

그다음 EMA = (0.9 × 90) + (0.1 × 0) = 81
<br>
<br>

### - 추가 학습 - Adam을 이해하기 위해

- Second Uncentered Moment = $g^2$
기울기의 제곱($g^2$) = Second Uncentered Moment <br>
딥러닝의 최적화 과정에서 gradient의 평균은 보통 0에 수렴한다고 가정하기 때문에, 기울기의 제곱은 곧 기울기의 분산(Variance)을 의미하게 된다. 즉, '방향(+, -)'은 무시하고 '최근 기울기가 얼마나 크게 요동치고 있는가(변동성, Magnitude)'를 측정하는 지표를 의미한다.

- EWMA (Exponentially Weighted Moving Average) = $E[g^2]$
수식에서 쓰인 $0.9$와 $0.1$의 가중치 합을 통해, 최근 데이터에는 큰 가중치(0.1)를 주고 과거 데이터에는 지수적으로 감쇠하는 가중치(e.g. $0.1 \times 0.9$, $0.1 \times 0.9^2$ ...)를 곱해 더한다. 이를 공식 용어로 지수 가중 이동 평균(EWMA)이라고 부르며, 수식에서는 기울기 제곱에 대한 기댓값 $E[g^2]$를 추정하는 과정으로 이해하면 된다.

- RMS = $\sqrt{E[g^2]}$
앞서 구한 EWMA 값에 최종적으로 제곱근(루트, $\sqrt{}$)을 씌운다.
어떤 값들을 제곱해서 평균을 낸 뒤 다시 루트를 씌우는 것을 RMS(Root Mean Square)라고 부르며, 이는 +와 -를 오가는 데이터의 Effective Magnitude를 나타낼 때 사용한다.

<br>

---

## 6. Adam

### - First Moment

momentum과 완전히 같은 수식이다.

- Momentum

Momentum은 SGD의 문제점(Saddle point, local minima, bouncing out of the valley, and jittering)을 해결하기 위한 방법이다. <br>
속도항을 추가하여 문제들을 해결하였다. 
$$
v_{t+1} = \rho \times v_t + \nabla f(x_t)
$$

여기서 $\rho$ 는 얼마나 많은 momentum을 가질지 결정하는 hyper parameter로, $\rho$ 가 커지면 새로운 속도 $v_{t+1}$ 은 현재 속도 $v_t$ 에 더 의존하게 된다. 이렇게 속도항을 통해 누적해준 후 weight 업데이트한다.
$$
w_{t+1} = w_t + \alpha \times v_{t+1}
$$

- First Moment

$$
m_t = \beta \times m_{t-1} + (1 - \beta) \times g_t
$$

헷갈리지 말 점은 $\beta$ 1을 사용한다. 
<br>

### - Second Moment

위에서 봤던 EMA와 완전히 같은 수식이다. 

- EMA
$$
\text{MeanSquare}(w,t) = \beta \times \text{MeanSquare}(w,t-1) + (1-\beta) \times g_t^2
$$

- Second Moment
$$
v_t = \beta \times v_{t-1} + (1 - \beta) \times g_t^2
$$

헷갈리지 말 점은 $\beta$ 2를 사용한다. <br>
즉 Adam은 Momentum과 RMSProp의 결합으로 구성된 Optimizer로, 같은 방향으로 계속 이동하면 점차 속도가 붙게 만든다.

그런데 **문제가 발생**한다. <br>
$\beta$ 1, $\beta$ 2는 1에 매우 가까운 값으로 보통 초기화된다. (0.9나 0.999같은)
그리고 설계에서 first_moment = 0, second_moment = 0으로 초기화한다면, 분모가 0이 되어 초기값이 매우 커지게 된다. 

$$
w_t = w_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**예시** <br>
$v_1 = (0.999 \times 0) + (1 - 0.999) \times g_1^2$ 이 되어서 $v_1 = (0.001) \times g_1^2$ 이 된다. <br>
$\beta 2 = 0.999$ 의 원래 의도는 과거 데이터의 비중을 99.9퍼센트로 유지하고, 새 데이터는 0.1퍼센트만 반영하는 것이었지만 초기 시작 시점에는 과거 데이터( $v_0$ )가 0이므로 새로 들어온 기울기 제곱( $g_1^2$ )에 $0.001$ 이라는 극단적으로 작은 비율만 적용되어 $v_1$ 값이 0에 매우 가까운 작은 수로 편향된 출력이 나온다. <br> $m_1$ 에서도 마찬가지이다.

Adam에서는 이 문제를 해결하기 위해 **Bias Correction**을 적용한다.

### - Bias Correction

- second moment
$$
v_t = (1-\beta_ 2) \sum_{i=1}^{t} \beta_ 2^{\,t-i}\, g_i^2
$$


양변에 기댓값을 취해서, $\mathbb{E}[v_t]$ 가 second moment $\mathbb{E}[g_t^2]$ 와 어떤 관계인지 확인해보자. <br>
여기서 기댓값을 취한다는 건, second moment가 일정하다 가정하는 것이다. <br> 
$\mathbb{E}[g_i^2] = \mathbb{E}[g^2]$

<br>

$$
\begin{aligned}
\mathbb{E}[v_t] &= \mathbb{E}\Big[(1-\beta_2) \sum_{i=1}^{t} \beta_2^{\,t-i}\, g_i^2\Big] \\
&= \mathbb{E}[g_t^2] \cdot (1-\beta_2) \sum_{i=1}^{t} \beta_2^{\,t-i} + \zeta \\
&= \mathbb{E}[g_t^2] \cdot (1-\beta_2^{\,t}) + \zeta
\end{aligned}
$$

둘째 줄은 과거의 $g_i^2$ 를 모두 현재의 $g_t^2$ 로 바꿔 쓴 것으로 그 차이에서 생기는 오차를 $\zeta$ 라고 한다. <br>
셋째 줄은 등비수열의 합이다. $(1-\beta_2)\sum_{k=0}^{t-1}\beta_2^k = 1-\beta_2^t$ <br>
<br>

따라서 $(1-\beta_2^{\,t})$ 라는 배율 차이가 있다는 걸 알 수 있다. 논문에서는 이 항이 0으로 초기화했기 때문에 생긴 것이라고 설명하고, 그래서 이 값으로 나눠서 보정한다.

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}
$$

### 왜 $(1-\beta_2^{\,t})$ 가 0으로 초기화해서 생겼다는 걸까?

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2
$$

<br>

$v_0$ 를 남긴 채 점화식을 풀면 다음과 같다. 

<br>

$$
v_t = \underbrace{\beta_2^{\,t}\, v_0}_{\text{초기값의 몫}} + \underbrace{(1-\beta_2)\sum_{i=1}^{t} \beta_2^{\,t-i}\, g_i^2}_{\text{실제 기울기들의 몫}}
$$

<br>

여기서 기댓값을 취하면, 

<br>

$$
\mathbb{E}[v_t] = \beta_2^{\,t}\, v_0 + (1-\beta_2^{\,t})\,\mathbb{E}[g^2]
$$

| 초기값 | $\mathbb{E}[v_t]$ | 치우침 |
| --- | --- | --- |
| $v_0 = \mathbb{E}[g^2]$ | $\mathbb{E}[g^2]$ | 없음 |
| $v_0 = 0$ | $(1-\beta_2^{\,t})\,\mathbb{E}[g^2]$ | $(1-\beta_2^{\,t})$ 배율 생김 |

<br>

따라서 $(1-\beta_2^{\,t})$ 가 0으로 초기화해서 생겼다고 추론할 수 있다.

<br>

**실제로 bias correction이 효과 있었나?**

- section 5
: momentum을 붙인 RMSProp에는 bias corection이 없다. 그래서 $\beta_ 2$ 가 1에 가까우면 초반 step이 매우 커지고 발산할 수 있는데, 이는 CIFAR 실험에서 RMSprop이 첫 epoch에 무너진 것으로 확인 할 수 있었다.

- section 6.4
: VAE 실험에서 bias correction을 뺀 경우와 넣은 경우를 비교한다. $\beta_ 2$ 가 1에 가까울 때 bias correction이 없으면 학습이 불안정해지는 것을 볼 수 있다. 


#### 초반에만 작용해야하는데 정말 그런가? 

$t$ 가 커지면 $\beta^t \to 0$ 이 되어 보정 계수가 1에 가까워지므로, 보정은 학습 초반에만 작동하게 된다.

따라서 최종적인 Adam의 수식은 다음과 같다.

$$
w_t = w_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

<br>

### Algorithm

hyper parameter: $\alpha = 0.001$, $\beta_ 1 = 0.9$, $\beta_ 2 = 0.999$, $\epsilon = 10^{-8}$

$$
\begin{aligned}
g_t = \nabla_\theta f_t(\theta_{t-1}) \\
m_t = \beta_ 1 m_{t-1} + (1-\beta_ 1)\, g_t \\
v_t = \beta_ 2 v_{t-1} + (1-\beta_ 2)\, g_t^2 \\
\hat m_t = m_t / (1-\beta_ 1^t), \qquad \hat v_t = v_t / (1-\beta_ 2^t) \\
\theta_t = \theta_{t-1} - \alpha \cdot \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$


---

## 7. AdamW

### - Adam의 한계

## 정리

- **RProp**: gradient의 부호만 사용하고, weight마다 step size를 조절. full-batch 학습에 적합.
- **RMSProp**: gradient 제곱의 지수 이동 평균으로 나누어, rprop의 아이디어를 mini-batch에서도 쓸 수 있게 만든 방법.
- **Adam**: Momentum + RMSProp으로, Bias Correction을 적용한 방법.

## 구현 및 실험 설계 

### 구현

```python
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

```python
class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

```python
class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

```python
class AdamW:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            params[key] -= self.lr * self.weight_decay * params[key]

            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

### 참고 논문: Empirical Performance Comparison of Optimization Algorithms for Machine Learning/Deep Learning (2024)

### 실험 설계
- 실험 A: 함수를 통한 성능 평가 (f1, f2 설정)
$$
f1(x,y) = x^2 + y^2 + 5\sin(x + y^2)
$$

$$
f2(x,y) = x^2 + y^2 + 5\sin(x + 2y^2)
$$

objective function을 해당 함수로 설정한 이유는 local minima가 많기 때문이다. f2의 경우, y축 방향으로 압축하였으므로 f1보다 f2에서 y방향의 gradient에서는 2배 그리고 gradient^2에서는 4배 더 커졌다.

즉 민감도를 확인할 수 있다.

- 실험 B: CIFAR-10 데이터셋을 통한 성능 평가 
