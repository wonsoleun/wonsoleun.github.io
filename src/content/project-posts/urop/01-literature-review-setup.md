---
title: "Literature Review and Experimental Setup"
order: 1
date: "2026-11-17"
description: "PushT Diffusion Policy 연구의 선행연구 정리와 실험 설정"
---

# 자료조사

- **본 연구 주제:** PushT 환경에서 시연 수와 초기 상태 다양성이 Diffusion Policy의 조작 성능 및 일반화에 미치는 영향 분석
- **이번 주차 목표 :** Diffusion Model의 기본 원리와 Diffusion Policy의 action chunk 생성 방식을 이해하고, 논문 읽기

<br>

---

## 1. 논문



| 순서 | 자료 | 읽는 범위 | 목적 |
| --- | --- | --- | --- |
| 1 | Lilian Weng, [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) | 첫 섹션(forward · reverse · 학습 loss)과 "Speed up Diffusion Models"의 DDIM 부분 | Diffusion 원리를 글과 그림으로 먼저 잡기 |
| 2 | Ho et al., [DDPM](https://arxiv.org/abs/2006.11239) (NeurIPS 2020) | 2장(Background), 3장(학습 목표), Algorithm 1 · 2 | 노이즈 예측 학습과 샘플링 절차를 원문으로 확인 |
| 3 | Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137) (RSS 2023 / IJRR 2024) | **정독.** 아래 2.1의 섹션 안내 참고 | 연구에서 쓰는 모델과 PushT 과제의 원논문 |
| 4 | Lin et al., [Data Scaling Laws](https://arxiv.org/abs/2410.18647) (ICLR 2025) | 1장, 3장, 4장, 7장. Fig. 4와 Fig. 7. 아래 2.2 참고 | "양보다 다양성"이라는 연구 출발점 |
| 추가 | Diffusion Policy 저자의 [state 기반 PushT Colab](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing) | 코드를 위에서 아래로 읽기 (실행은 선택) | 논문 내용이 코드에서 어떻게 구현되는지 확인 |

<br>

---

## 2. 집중해서 공부할 부분

<br>

### 2.1 Diffusion Policy

**집중해서 읽을 부분**

|  | 내용 | 본 연구와의 연결 |
| --- | --- | --- |
| 2장 Diffusion Policy Formulation | DDPM을 행동 생성에 적용하는 방법. 관측을 조건으로 행동 시퀀스를 denoising | 모델의 기본 원리 |
| 3.1 Network Architecture | CNN 기반과 Transformer 기반 | 본 연구는 CNN(1D U-Net) 사용 |
| 3.3 Noise Schedule, 3.4 Accelerating Inference | Square Cosine Schedule, DDIM으로 추론 가속 | 본 연구의 "학습 DDPM, 추론 DDIM"의 근거 |
| 4.3 Benefits of Action-Sequence Prediction | action chunk를 쓰는 이유 | `horizon`, `n_action_steps` 이해 |
| 5.1–5.4 Evaluation | 시뮬레이션 과제, 평가 방법, ablation | PushT 설정과 평가 방식 |
| Appendix A.1, A.4, A.5 | 정규화, 하이퍼파라미터, **Data Efficiency** | A.5는 RQ1(시연 수)과 직접 관련 |
| Appendix B.1 | Observation horizon ablation | `n_obs_steps` = 2의 근거 |

<br>

**가볍게 읽을 부분:** 3.2(Visual Encoder), 6–7장(실제 로봇 실험), Appendix C–D. 본 연구는 state 입력과 시뮬레이션만 다룬다.

<br>

**읽으면서 확인할 용어 (논문 표기 <-> LeRobot 표기)**

| 논문 | LeRobot config | 뜻 | 본 연구에서 설정 값 |
| --- | --- | --- | --- |
| T_o (observation horizon) | `n_obs_steps` | 입력으로 쓰는 최근 관측 step 수 | 2 |
| T_p (prediction horizon) | `horizon` | 한 번에 예측하는 action step 수 | 16 |
| T_a (action execution horizon) | `n_action_steps` | 예측 중 실제로 실행하는 step 수 | 8 |

<br>

### 2.2 Data Scaling Laws

**집중해서 읽을 부분:** 1장(서론), 3장(Approach), 4장(Data Scaling Laws), 7장(한계)

<br>

**집중할 내용**
- 환경 수(M), 물체 수(N), 환경-물체 쌍당 시연 수(K)를 따로 바꿔 가며 비교한 실험 설계. 본 연구에서 시연 수와 선택 방식을 분리하는 것과 같은 구조다.
- Fig. 4: 환경-물체 쌍이 늘수록 성능이 계속 오르고, 시연 비율에 따른 차이는 작다.
- Fig. 7 오른쪽: 쌍 수를 고정하면 총 시연 수가 늘어도 성능이 포화된다.
- 논문은 쌍당 시연 약 50개를 넘으면 추가 효과가 작다고 보고한다.

<br>

**가볍게 읽을 부분:** 5장(검증 실험), 6장(모델 크기), appendix

<br>

### 2.3 DDPM

2장과 3장, Algorithm 1(학습)과 Algorithm 2(샘플링) 위주로 읽는다. 4장 이후의 이미지 생성 실험은 가볍게 읽어도 된다.

<br>

---

## 3. 원논문 설정과 비교

Diffusion Policy 원논문과 공식 저장소를 직접 확인해, 본 연구의 설정과 같은 점과 다른 점을 나눴다.


### 3.1 원논문과 같은 점

- **Horizon 설정:** Diffusion Policy 논문 Appendix A.4에서 PushT는 T_o = 2, T_a = 8, T_p = 16이다.
- **정규화:** 논문 Appendix A.1은 각 차원을 min-max로 [-1, 1]에 맞춘다. 평균 0 · 분산 1 정규화를 쓰지 않는 이유는 DDPM이 매 step 예측값을 [-1, 1]로 자르기 때문이다. 
- **DDIM 추론:** 논문의 실제 로봇 실험은 학습 100 iteration, 추론 10 iteration의 DDIM을 쓴다. 본 연구의 "학습 DDPM 100 step, 추론 DDIM 10 step"은 원논문에서 이미 쓰인 방식이다.
- **Observation horizon:** Appendix B.1에 따르면 state 기반 Diffusion Policy는 observation horizon에 민감하지 않다.

<br>

### 3.2 원논문과 다른 점

|  | 논문 · 저장소 | 본 연구 | 처리 |
| --- | --- | --- | --- |
| 시연 수 | 논문 Table 3은 PushT 시연 200개로 표기 | 데이터셋 기준 206개 | 데이터셋을 내려받아 실제 에피소드 수를 세어 확인. 본문에는 데이터셋 기준 수를 쓴다 |
| keypoint 수 | 논문은 블록 keypoint 9개로 설명 | LeRobot 데이터셋은 16차원(8개) | 데이터셋 카드와 공개 checkpoint config에서 16차원임을 확인. 본문에는 데이터셋 기준으로 쓴다 |
| 평가 지표 | 논문은 PushT에 target area coverage를 사용 | success rate + max coverage | 논문의 PushT 수치와 내 success rate를 직접 비교하지 않는다 |
| checkpoint 선택 | 공식 저장소 README: 논문 수치는 `max`, `k_min_train_loss` 집계값 | 마지막 checkpoint 평가 | 내 수치가 논문보다 낮게 나와도 집계 방식 차이일 수 있다 |
| 공개 checkpoint 점수 | 공식 저장소의 lowdim 예시 checkpoint는 평균 점수 약 0.92–0.97 | 참고용으로만 사용 | coverage 점수이므로 success rate와 다른 값이다 |

<br>

---

## 4. 스스로 대답할 수 있어야 함

**Diffusion 기본**
1. forward process와 reverse process는 각각 무엇을 하는가?
2. 모델은 학습할 때 무엇을 예측하고, loss는 무엇인가?
3. DDPM 샘플링과 DDIM 샘플링은 무엇이 다르며, DDIM은 왜 더 적은 step으로 가능한가?
4. 학습은 100 step으로 하고 추론은 10 step으로 해도 되는 이유는 무엇인가?

**Diffusion Policy**
5. 이미지 생성 diffusion과 비교해, Diffusion Policy에서 "생성 대상"과 "조건"은 각각 무엇인가?
6. 행동을 한 step씩이 아니라 16 step 묶음(action chunk)으로 예측하는 이유는 무엇인가?
7. 16 step을 예측하고 8 step만 실행하는 이유는 무엇인가? 실행 길이를 늘리거나 줄이면 무엇이 달라지는가?
8. PushT의 state 입력은 무엇으로 구성되는가? 성능은 어떻게 측정하는가?
9. 평균 0 · 분산 1이 아닌 min-max 정규화를 쓰는 이유는 무엇인가?

**Data Scaling Laws**
10. 이 논문에서 "다양성"은 무엇을 뜻하며, 내 연구의 "초기 상태 다양성"과 어떻게 다른가?
11. Fig. 4에서 시연 비율이 달라도 곡선이 겹친다는 것은 무엇을 의미하는가?
12. 이 논문의 결과를 내 연구에 그대로 적용할 수 없는 이유 두 가지는 무엇인가?

<br>

---

## 5. 환경 설정

|  | 설정 | 비고 |
| --- | --- | --- |
| 운영체제 | Ubuntu 22.04 LTS (네이티브 또는 Windows 11 + WSL2) | VirtualBox VM은 GPU를 쓸 수 없어 학습에 부적합. |
| GPU | NVIDIA, VRAM 8GB 이상 권장 | 기본 PyTorch wheel은 NVIDIA 드라이버 580.65 이상 필요. 낮으면 cu126 wheel을 먼저 설치 |
| Python | 3.12 (conda 가상환경) |  |
| PyTorch / CUDA | LeRobot 설치 시 함께 설치되는 PyTorch 2.x (CUDA 빌드) | 실제 버전 (확인 예정) |
| policy 구현 | LeRobot (Hugging Face) `diffusion` policy | commit hash (확인 예정). 실험 기간 중 업데이트하지 않음 |
| 시뮬레이터 | gym-pusht (`PushT-v0`, Pymunk 2D 물리 엔진) | `pip install 'lerobot[pusht]'`로 함께 설치 |
| 버전 관리 | 코드는 Git으로 관리, `conda env export` · `pip freeze` 결과 저장 | 모든 run의 config · seed · subset 목록 · commit hash를 함께 저장 |

<br>

---

## 6. 데이터

|  | 내용 |
| --- | --- |
| 데이터셋 | `lerobot/pusht_keypoints` (Hugging Face) |
| 원본 | Diffusion Policy 저자 공개 데이터 `pusht_cchi_v7_replay.zarr` |
| 규모 | 시연 206개(에피소드), 25,650 프레임, 10 fps |
| 파일 형식 | Parquet. LeRobot dataset 형식 버전 (확인 예정: `meta/info.json`의 `codebase_version`) |
| 라이선스 | Apache-2.0 |

<br>

**필드**
 
| 필드 | 형식 | 사용 방법 |
| --- | --- | --- |
| `observation.state` | float32, 2차원 (agent x, y) | policy 입력. 첫 프레임 값은 initial state feature로 사용 |
| `observation.environment_state` | float32, 16차원 (블록 keypoint 8개의 x, y) | policy 입력. 첫 프레임 값은 initial state feature로 사용 |
| `action` | float32, 2차원 (agent 목표 위치 x, y) | policy 출력(학습 target) |
| `episode_index`, `frame_index` | int64 | 에피소드 단위 subset 구성, 첫 프레임(`frame_index = 0`) 식별 |
| `next.reward`, `next.done`, `next.success` | float32, bool, bool | 학습에는 쓰지 않음 |
| `timestamp`, `index` | float32, int64 | 학습에는 쓰지 않음 |
 
**주의**
- image 버전인 `lerobot/pusht`의 `observation.state`는 agent 위치 2차원뿐이라 블록 정보가 없다. state 기반 policy에는 keypoints 버전을 쓴다.
- Diffusion Policy 원논문은 블록 keypoint를 9개, 시연을 200개로 적고 있다. 본 연구는 데이터셋 기준인 keypoint 8개(16차원), 시연 206개로 쓴다.

<br>

---

## 7. 모델 설정

: 공개 checkpoint `lerobot/diffusion_pusht_keypoints`의 config를 그대로 따르며, 모든 실험 조건에 동일하게 적용한다.

|  | 값 | 비고 |
| --- | --- | --- |
| Policy | Diffusion Policy, Conditional 1D U-Net (temporal CNN) | `down_dims` 512 · 1024 · 2048, `kernel_size` 5, FiLM scale modulation 사용 |
| 입력 | `observation.state` 2차원 + `observation.environment_state` 16차원, 최근 2 step (`n_obs_steps` = 2) | 관측을 encoding해 U-Net의 global conditioning으로 사용 |
| 출력 | 2차원 action 16개 (`horizon` = 16) | 그중 앞 8개를 실행한 뒤 다시 예측 (`n_action_steps` = 8) |
| 학습 noise schedule | 100 timestep, `squaredcos_cap_v2`, epsilon 예측 | `num_train_timesteps` = 100 |
| 추론 sampler | DDIM, 10 step | `noise_scheduler_type` = DDIM, `num_inference_steps` = 10 |
| 정규화 | state · action 모두 min-max | 통계는 전체 206개 기준으로 고정. subset 학습 시 LeRobot이 어느 통계를 쓰는지 (확인 예정) |
| 학습 step · batch size | 1k step당 소요 시간을 측정해 확정 (확인 예정) | 모든 조건에서 동일. 참고: 공개 checkpoint는 200,000 step, 약 5시간 |
| 기타 하이퍼파라미터 | 공개 checkpoint config 값 | `diffusion_step_embed_dim` 128, `clip_sample` true. config 파일을 그대로 보관 |

<br>

---

## 8. 평가 설정과 재현 정보
 
|  | 내용 |
| --- | --- |
| 평가 initial state | 환경 seed 0–99로 만든 100개, 모든 policy에 동일하게 사용 |
| 에피소드 길이 | 최대 300 step |
| 성공 기준 | 블록과 목표 영역의 최대 겹침(max overlap) 0.95 이상 |
| 평가 지표 | success rate, max coverage |
| 학습 seed | 조건당 3개 |
| checkpoint | 마지막 checkpoint를 평가 (best checkpoint를 고르지 않음) |
| 기록 항목 | LeRobot commit hash, 데이터셋 버전, config 파일, subset 에피소드 목록 |

**주의**
- 공개 checkpoint의 success rate가 71.0%(500 에피소드, 모델 카드 기준)이므로, 시연 206개 전체로 학습한 baseline도 100%에 가깝게 나오지 않을 수 있다.
- 공개 checkpoint는 200,000 step 학습에 약 5시간이 걸렸다. 같은 step 수로 18회 학습하면 90시간 이상이 걸리므로, 학습 step 수는 소요 시간을 직접 잰 뒤 정한다.
