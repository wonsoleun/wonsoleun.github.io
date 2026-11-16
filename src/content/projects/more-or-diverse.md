---
title: "More or Diverse?"
description: "An Empirical Study on the Effects of Demonstration Quantity and Initial-State Diversity on Diffusion Policy in the PushT Environment."
tags:
  - "Imitation Learning"
  - "Diffusion Policy"
  - "Agent"
---

## Overview

1.  공개 PushT 데이터셋의 시연 206개로 state 기반 Diffusion Policy를 학습해 baseline 성능을 재현한다.
2.  시연 수(20·50·100·206개)와 데이터 선택 방식(random sampling vs diversity-aware selection)을 바꿔 학습하고, 고정된 initial state 100개에서 success rate를 비교한다.
3.  평가 initial state와 학습 데이터 initial state 사이의 거리에 따라 성능이 얼마나 떨어지는지 측정해, policy가 일반화하는지 학습 데이터를 기억해 재사용하는지 분석한다. 