---
title: "VAE: Variational Auto-Encoder"
date: "2026-11-25"
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

$$\text{loss} = \underbrace{\text{다시 그린 그림이 원본과 같은가}}_{\text{Reconstruction}} + \underbrace{z\text{가 } N(0, I) \text{ 근처에 있는가}}_{\text{KL}}$$

---

## 2. 용어와 기호

| 기호 | 이름 | VAE에서의 역할 |
| --- | --- | --- |
| `x` | 데이터 | 그림 |
| `z` | latent variable | 숨은 설정값 |
| `p(z)` | prior | N(0, I)로 정해 둠 |
| `p_θ(x\|z)` | likelihood | decoder. z로 x를 만듦 |
| `p(x)` | marginal likelihood (evidence) | 데이터가 나올 확률. 최대화하고 싶은 것 |
| `p(z\|x)` | posterior | x를 봤을 때 진짜 z의 분포 |
| `q_φ(z\|x)` | approximate posterior | encoder, p(z\x)를 따름 |
| `p(x, z)` | joint distribution | p_θ(x\z) · p(z) |

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

<br>

### 3.2 해석해보자

- **KL ≥ 0** 이므로 **ELBO ≤ log p(x)**. 그래서 이름이 Evidence **Lower** BOund다.
- `log p(x)`는 `q`와 무관하다. 따라서 **ELBO를 키우면 KL이 줄어든다.** 즉 `q`가 True posterior에 가까워진다.
- `q`가 True posterior와 같아지면 KL = 0이고 ELBO = log p(x)다.
- **ELBO를 최대화**하는 게 VAE 학습

### q(encoder)를 어떻게 바꿔도 log p(x)의 값은 변하지 않는다.

$$p_\theta(x) = \int p_\theta(x|z)\,p(z)\,dz$$

<br>

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

<br>

### 4.1 Decoder의 parameter θ는 무엇을 학습할까?

Decoder는 단순히 `z`를 받아 이미지 픽셀을 바로 복사하는 것이 아니라, $p_\theta(x|z)$ 를 학습한다.

$$p_\theta(x|z)$$

- `z`: encoder가 만든 latent vector
- `p_θ(x|z)`: 이 `z`가 주어졌을 때 어떤 원본 데이터 `x`가 생성될 수 있는지를 나타내는 확률분포
- `θ`: `z`를 받아 이 확률분포를 출력하도록 decoder를 조정하는 신경망 파라미터

따라서 decoder가 학습하는 것은 **latent vector `z`를 데이터 `x`로 생성하는 방법**이다. 정확히는 `x` 자체 하나만 출력하도록 외우는 것이 아니라, `z`가 주어졌을 때 데이터가 어떤 분포를 따를지 모델링한다.



**학습 과정**

1. 원본 이미지 `x`를 encoder에 넣어 `q_φ(z\|x)`의 파라미터를 구한다.
2. Reparameterization Trick으로 latent vector `z`를 샘플링한다.
3. decoder가 `z`를 입력받아 `p_θ(x|z)`를 출력한다.
4. 원본 `x`가 decoder가 출력한 분포에서 높은 확률을 갖도록 Reconstruction Loss를 계산한다.
5. 이 손실의 gradient로 decoder의 `θ`를 업데이트한다. 즉, 같은 `z`가 들어왔을 때 원본 `x`를 더 잘 설명하는 분포를 출력하도록 학습한다.

<br>

여기서 Reconstruction Loss는 다음과 같다.

$$\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

이 항에는 `θ`가 들어 있으므로 decoder의 파라미터가 업데이트된다. 반면 KL 항은 `q_φ(z|x)`와 고정된 prior `p(z)`를 비교하므로 직접적으로는 encoder의 파라미터 `φ`에 영향을 준다.

| 파라미터 |  | 무엇을 학습하는가? | 주로 영향을 받는 Loss |
| --- | --- | --- | --- |
| `θ` | Decoder | z로부터 p_θ(x\z)를 생성하는 방법 | Reconstruction |
| `φ` | Encoder | 입력 x가 주어졌을 때 latent distribution q_φ(z\x)를 추정하는 방법 | Reconstruction + KL |



즉, `θ`는 latent vector가 원본 데이터를 잘 설명하도록 decoder를 학습시키는 파라미터이고, `φ`는 입력 데이터를 적절한 latent distribution으로 표현하도록 encoder를 학습시키는 파라미터다. 두 네트워크가 함께 학습되면서 유용한 latent space와 생성 과정이 만들어진다.

<br>

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

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

<br>

### 해결

random node를 바깥으로 빼낸다. → $\epsilon$ 추가

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

- 이제 `z`는 μ, σ에 대한 덧셈과 곱셈이라 미분 가능
- random node인 ε은 학습 대상이 아니므로 미분할 필요가 없고 상수 취급하면 되니까.
- 분포는 그대로다. `μ + σε`는 여전히 `N(μ, σ²)`를 따른다.

$$
\mu + \sigma \epsilon \sim \mathcal{N}(\mu, \sigma^2)
$$

**핵심** = sampling에서 미분 가능하도록 해서 encoder까지 gradient가 전달되도록 했다.

$$
\frac{\partial z}{\partial \mu} = 1
\qquad
\frac{\partial z}{\partial \sigma} = \epsilon
$$

```python
import torch

def reparameterize(mu, sigma):
    eps = torch.randn_like(mu)  # random noise 설정, 평균 0, 분산 1인 표준정규분포에서 무작위 값
    return mu + eps*sigma  # Reparameterization Trick, latent variable (z) 출력

x = torch.randn((1, 1, 28, 28)) # 랜덤 이미지 (Batch, Channels, Height, Width)
mu, sigma = vae_encoder(x)  # encoder에서 출력된 평균과 분산
z = reparameterize(mu, sigma)  # Reparameterization Trick를 통한 샘플링 
```


---

## 6. DDPM과의 연결

DDPM은 encoder가 고정되어 있고, latent variable이 T단계의 Markov chain으로 이어진 hierarchical VAE다.

| | VAE | DDPM |
| --- | --- | --- |
| 데이터 | `x` | `x_0` |
| latent variable | `z` 하나 | `x_1, …, x_T` |
| encoder | `q_φ(z\|x)`, **학습함** | forward `q(x_t\|x_{t−1})`, noise 넣기, **고정** |
| decoder | `p_θ(x\|z)`, 한 번에 reconstruction | reverse `p_θ(x_{t−1}\|x_t)`, **T번 나눠** reconstruction |
| prior | `N(0, I)` | `p(x_T) = N(0, I)` |

DDPM의 Forward process `q`는 정해진 noise schedule에 따라 작동하므로 학습 가능한 Encoder parameter가 없다. 반면 Reverse process `p_θ`는 noise를 제거하는 방법을 학습한다.

<br>

같은 방식으로 ELBO를 쓰면 DDPM loss의 세 항이 나온다.

$$-\log p_\theta(x_0) \le \mathbb{E}_q\Big[\underbrace{D_{KL}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T:\ \text{VAE의 KL 항}} + \sum_{t=2}^{T}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))}_{L_{t-1}:\ \text{VAE에 없는 항}} \underbrace{-\log p_\theta(x_0|x_1)}_{L_0:\ \text{VAE의 reconstruction 항}}\Big]$$

- **L_T:** encoder가 고정이라 학습할 것이 없는 상수 → 무시
- **L_0:** Reverse process의 마지막 단계인 `x_1` → `x_0`에서 원본 데이터가 잘 생성되는지 측정
- **L_{t−1}:** 두 분포가 가까워지도록 Reverse process의 parameter `θ`를 학습

$$
L_{t-1} = D_{KL}\big(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)\big)
$$

- $q(x_{t-1}|x_t,x_0)$ : 원본 x_0를 알고 있을 때 계산할 수 있는, 실제 한 단계 전 상태의 posterior 분포 
- $p_\theta(x_{t-1}|x_t)$ : DDPM이 현재 x_t를 보고 예측하는 한 단계 전 상태의 분포

| VAE Loss | DDPM | |
| --- | --- | --- |
| Reconstruction Loss: `-E_q[log p_θ(x\|z)]` | `L_0: -log p_θ(x_0\|x_1)` | 최종적으로 원본 데이터 복원 |
| KL Loss: `KL(q_φ(z\|x) \| p(z))` | `L_T: KL(q(x_T\|x_0) \| p(x_T))` | 마지막 latent distribution과 prior 비교 |
| 별도 항 없음 | `Σ L_{t-1}` | 중간 Reverse step마다 noise를 제거하는 방법 학습 |

VAE에서는 한 번에 latent variable을 거쳐 했지만, DDPM에서는 reconstruction을 T번에 나눠 조금씩 denoising으로 바꾼다.