---
title: "Preliminary Research"
order: 1
date: "2026-11-13"
description: "Reviewing related research and academic papers."
---

## 선행 연구

| 논문 | 핵심 내용 | 본 연구와의 관계 |
| --- | --- | --- |
| [1] Chi et al., Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023 / IJRR 2024) [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | denoising으로 action chunk를 생성하는 policy, PushT 벤치마크 제시 | 기반 모델과 실험 환경 |
| [2] Lin, Hu et al., Data Scaling Laws in Imitation Learning for Robotic Manipulation (ICLR 2025) [arXiv:2410.18647](https://arxiv.org/abs/2410.18647) | 실제 로봇에서 일반화 성능이 환경·물체 수와 power-law 관계; 환경당 시연 수는 일정 수준 이후 효과 작음 | "양보다 다양성" 주장을 단일 2D 과제의 initial state 다양성으로 옮겨 검증 |
| [3] He et al., Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives (ICLR 2026) [arXiv:2505.05787](https://arxiv.org/abs/2505.05787) | 소규모 데이터에서 Diffusion Policy가 학습 action chunk를 꺼내 쓰는 lookup table처럼 동작 | 거리별 success rate 분석(RQ3)의 근거 |
| [4] Ho et al., Denoising Diffusion Probabilistic Models (NeurIPS 2020) [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | 노이즈 추가(forward), 제거(reverse) 과정, 학습 목표(노이즈 예측) | Diffusion Policy 이해를 위한 배경지식 |


---

## 환경 설정

| 항목 | 설정 |
| --- | --- |
| 운영체제 | Ubuntu 22.04 LTS (네이티브 또는 Windows 11 + WSL2) |
| 대체 환경 | Google Colab (GPU 런타임) |
| GPU | NVIDIA, VRAM 8GB 이상 권장 |
| Python | 3.10 (conda 가상환경) |
| PyTorch / CUDA | PyTorch 2.x, CUDA 12.x (GPU 드라이버와 호환되는 버전) |
| policy 구현 | LeRobot (Hugging Face) `diffusion` policy |
| 시뮬레이터 | gym-pusht (`PushT-v0`, Pymunk 2D 물리 엔진) |
| 버전 관리 | 코드는 Git으로 관리, `conda env export`·`pip freeze` 결과 저장 |


---

## 데이터

- 데이터셋 = lerobot/pusht_keypoints (Hugging Face)

| 항목 | 값 | 비고 |
| --- | --- | --- |
| Policy | Diffusion Policy, Conditional 1D U-Net (temporal CNN) | LeRobot pusht_keypoints 기본 config |
| 입력 | `observation.state` 2D + `observation.environment_state` 16D, 최근 2 step (`n_obs_steps` = 2) | 관측을 encoding해 U-Net의 global conditioning으로 사용 |
| 출력 | 2D action 16개 (`horizon` = 16) | 그중 앞 8개를 실행한 뒤 다시 예측 (`n_action_steps` = 8) |
| 학습 noise scheduler | DDPM, diffusion step 100 | 공식 설정 유지 |
| 추론 sampler | DDIM, 10 step | 학습한 noise schedule은 그대로 두고 추론 step만 줄여 18개 policy × 100 에피소드의 평가 비용을 절감 |
| 정규화 | state·action min-max | 정규화 통계는 subset별이 아닌 전체 206개 기준으로 고정 |
| 학습 step·batch size | 3주차에 1k step당 소요 시간을 측정해 확정 | 모든 조건에서 동일 |
| 기타 하이퍼파라미터 | 공식 config 기본값 | config 파일을 그대로 보관 |
