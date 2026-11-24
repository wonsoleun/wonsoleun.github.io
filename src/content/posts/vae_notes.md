---
title: "VAE: Variational Auto-Encoder"
date: "2026-11-17"
description: "Paper reading notes"
author: "Soleun Won"
tags:
  - "Autoencoder"
---



**VAE 공부하는 이유:** DDPM과 Diffusion Policy의 loss는 VAE와 같은 방식(ELBO)으로 만들어진다. 때문에 VAE 수식을 잡고 가야 다음 논문들도 이해가 된다.


---

## 1. 수식 없이 먼저 이해하자

- 데이터 = MNIST
그림마다 latent variable `z` 가 있음. <br>
아무 `z`를 입력으로 받으면 그림을 그리는 게 **decoder**. <br>
근데 데이터에는 `z`가 적혀있지 않음. <br>
때문에 `z` 추측하는 **encoder** 필요함. encoder는 새로운 출력 `x'`가 원본 `x`와 같아지도록 학습함. <br>
여기서 `z`는 분포로(평균 μ, 분산 σ) (**reparameterization trick**)

→ loss의 첫 번째 항, **Reconstruction** <br>

이제 새로운 그림을 만들 때를 위해 `z`가 `N(0, I)` 근처에 있게 하는 게 필요함. <br>

→ loss의 두 번째 항, **KL (Regularization)**

```
loss = (다시 그린 그림이 원본과 같은가)  +  (z가 N(0, I) 근처에 있는가)
       └──── Reconstruction ────┘        └──────── KL ────────┘
```

---

## 2. 용어와 기호

| 기호 | 이름 | VAE에서의 역할 |
| --- | --- | --- |
| `x` | 데이터 | 그림 |
| `z` | latent variable | 숨은 설정값 |
| `p(z)` | prior | `N(0, I)`로 정해 둠 |
| `p_θ(x\|z)` | likelihood | **decoder**. `z`로 `x`를 만듦 |
| `p(x)` | **marginal likelihood (evidence)** | 데이터가 나올 확률. 최대화하고 싶은 것 |
| `p(z\|x)` | posterior | `x`를 봤을 때 진짜 `z`의 분포 |
| `q_φ(z\|x)` | approximate posterior | **encoder**. `p(z|x)`를 흉내 내는 신경망 |
| `p(x, z)` | joint distribution | `p_θ(x|z) · p(z)` |

θ는 decoder의 파라미터, φ는 encoder의 파라미터다.

**marginal likelihood**

$$p(x) = \int p_\theta(x|z)\,p(z)\,dz \qquad \text{(정의: } z\text{를 적분해서 없앰 = marginalize)}$$

$$p(x) = \frac{p(x, z)}{p(z|x)} \qquad \text{(곱셈 법칙 } p(x,z) = p(z|x)\,p(x)\text{을 정리한 항등식)}$$


---

## 3. ELBO 유도

### 3.1 핵심 등식

어떤 `q(z|x)`를 가져와도 다음이 **항상** 성립한다.

$$\log p(x) = \text{ELBO} + D_{KL}\big(q(z|x)\,\|\,p(z|x)\big)$$

**유도**

$$
\begin{aligned}
\log p(x) &= \mathbb{E}_{q(z|x)}[\log p(x)]\\
&= \mathbb{E}_{q}\left[\log \frac{p(x, z)}{p(z|x)}\right] && \text{항등식}\\
&= \mathbb{E}_{q}\left[\log \frac{p(x, z)}{q(z|x)} \cdot \frac{q(z|x)}{p(z|x)}\right] && q/q = 1\text{을 곱함}\\
&= \underbrace{\mathbb{E}_{q}\left[\log \frac{p(x, z)}{q(z|x)}\right]}_{\text{ELBO}} + \underbrace{\mathbb{E}_{q}\left[\log \frac{q(z|x)}{p(z|x)}\right]}_{D_{KL}(q(z|x)\,\|\,p(z|x))}
\end{aligned}
$$

### 3.2 해석해보자

- **KL ≥ 0** 이므로 **ELBO ≤ log p(x)**. 그래서 이름이 Evidence **Lower** BOund다.
- `log p(x)`는 `q`와 무관한 고정값이다. 따라서 **ELBO를 키우면 KL이 줄어든다.** 즉 `q`가 True posterior에 가까워진다.
- `q`가 True posterior와 같아지면 KL = 0이고 ELBO = log p(x)다.
- 계산할 수 없는 `log p(x)` 대신 **계산할 수 있는 ELBO를 최대화**한다. 이게 VAE 학습

### 3.3 다른 유도: Jensen 부등식

log는 concave function이라 Jensen 부등식에 의해서 `log E[X] ≥ E[log X]`가 성립한다.

$$\log p(x) = \log \int q(z|x) \frac{p(x,z)}{q(z|x)} dz = \log \mathbb{E}_q\left[\frac{p(x,z)}{q(z|x)}\right] \ge \mathbb{E}_q\left[\log \frac{p(x,z)}{q(z|x)}\right] = \text{ELBO}$$



---

## 4. ELBO를 두 항으로 쪼개기 → Loss

`p(x, z) = p_θ(x|z) p(z)`를 대입하고 log를 나눈다.

$$
\begin{aligned}
\text{ELBO} &= \mathbb{E}_{q}\left[\log p_\theta(x|z)\right] + \mathbb{E}_{q}\left[\log \frac{p(z)}{q_\phi(z|x)}\right]\\
&= \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)
\end{aligned}
$$

최대화를 (-)부호를 통해 최소화로 바꾸면,

$$\mathcal{L}(\theta, \phi; x) = \underbrace{-\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{Reconstruction error}} + \underbrace{D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{Regularization error}}$$

### 주의할 점

| | 3장의 KL | 4장의 KL |
| --- | --- | --- |
| 식 | `KL(q(z\|x) ‖ p(z\|x))` | `KL(q(z\|x) ‖ p(z))` |
| 비교 대상 | **True posterior** | **prior N(0, I)** |
| 역할 | ELBO와 log p(x)의 차이 | loss의 Regularization 항 |



---

## 5. Reparameterization Trick

### 문제

encoder는 `μ, σ`를 내놓는다. decoder에 입력으로 넣기 위해 `z ~ q_φ(z|x)`를 샘플링해야한다. 근데 샘플링은 미분할 수 없는 연산이라 encoder 파라미터 φ로 gradient가 전달되지 않는다.

### 해결

random node를 바깥으로 빼낸다. → $\epsilon$ 추가

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

- 이제 `z`는 μ, σ에 대한 덧셈과 곱셈이라 미분 가능
- random node인 ε은 학습 대상이 아니므로 미분할 필요가 없다.
- 분포는 그대로다. `μ + σε`는 여전히 `N(μ, σ²)`를 따른다.


```
import torch

def reparameterize(mu, sigma):
    eps = torch.randn_like(std)  # 평균이 0이고 분산이 1인 표준 정규 분포에서 무작위 샘플링
    return mu + eps*sigma  # Reparameterization Trick

x = torch.randn((1, 3, 28, 28)) # 랜덤 이미지 (Batch, Channels, Height, Width)
mu, sigma = vae_encoder(x)  # 인코더에서 출력된 평균과 분산
z = reparameterize(mu, sigma)  # Reparameterization Trick를 통한 샘플링
```


---

## 6. DDPM과의 연결

DDPM은 encoder가 고정되어 있고, latent variable이 T단계의 Markov chain으로 이어진 hierarchical VAE다.

| | VAE | DDPM |
| --- | --- | --- |
| 데이터 | `x` | `x_0` |
| 잠재 변수 | `z` 하나 | `x_1, …, x_T` |
| encoder | `q_φ(z\|x)`, **학습함** | forward `q(x_t\|x_{t−1})`, noise 넣기, **고정** |
| decoder | `p_θ(x\|z)`, 한 번에 reconstruction | reverse `p_θ(x_{t−1}\|x_t)`, **T번 나눠** reconstruction |
| prior | `N(0, I)` | `p(x_T) = N(0, I)` |

같은 방식으로 ELBO를 쓰면 DDPM loss의 세 항이 나온다.

$$-\log p_\theta(x_0) \le \mathbb{E}_q\Big[\underbrace{D_{KL}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T:\ \text{VAE의 KL 항}} + \sum_{t=2}^{T}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))}_{L_{t-1}:\ \text{VAE에 없는 항}} \underbrace{-\log p_\theta(x_0|x_1)}_{L_0:\ \text{VAE의 복원 항}}\Big]$$

- **L_T:** encoder가 고정이라 학습할 것이 없는 상수 → 무시
- **L_0:** 마지막 한 단계 reconstruction
- **L_{t−1}:** 각 단계의 "한 단계 되돌리기"가 정확한지. **DDPM 학습의 핵심**이며, 정리하면 노이즈 예측 MSE `‖ε − ε_θ(x_t, t)‖²`가 된다.

VAE의 한 번에 reconstruction을 T번에 나눠 조금씩 denoising으로 바꾼 것이 DDPM이다.
