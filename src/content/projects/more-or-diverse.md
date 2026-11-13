---
title: "More or Diverse?"
description: "An Empirical Study on the Effects of Demonstration Quantity and Initial-State Diversity on Diffusion Policy in the PushT Environment."
tags:
  - "Imitation Learning"
  - "Diffusion Policy"
image: "/images/placeholder.svg"
---

## Overview

1.  공개 PushT 데이터셋의 시연 206개로 state 기반 Diffusion Policy를 학습해 baseline 성능을 재현한다.
2.  시연 수(20·50·100·206개)와 데이터 선택 방식(random sampling vs diversity-aware selection)을 바꿔 학습하고, 고정된 initial state 100개에서 success rate를 비교한다.
3.  평가 initial state와 학습 데이터 initial state 사이의 거리에 따라 성능이 얼마나 떨어지는지 측정해, policy가 일반화하는지 학습 데이터를 기억해 재사용하는지 분석한다.

![Theseus mouse demonstration](/images/placeholder.svg)

## Technical Implementation

The system consisted of:

1. **The Mouse**: A small wheeled vehicle containing magnetic sensors
2. **The Maze**: A 5x5 grid with movable walls and hidden pathways  
3. **Control System**: Over 40 telephone relays that stored the learned path
4. **Power System**: Standard electrical power with relay-based logic

### How It Learned

1. **Exploration Phase**: Mouse moves randomly, exploring all paths
2. **Success Detection**: When reaching goal, path is electrically recorded
3. **Memory Storage**: Relays store which moves lead to success
4. **Execution**: Subsequent runs use stored path, ignoring dead ends

The relays used were "sticky" - once energized, they stayed that way until manually reset. This created permanent memory of successful paths.

## Innovations

- **First learning machine**: Demonstrated learning without explicit programming
- **Relay computation**: Pioneering use of telephone equipment for computation
- **Artificial intelligence**: Early AI before the field had a name

## Public Reception

The machine was demonstrated to the press and technical community, generating significant interest. It appeared in:

- Popular Science Monthly
- Scientific American
- Various news outlets

## Legacy

Theseus demonstrated that:
1. Machines can exhibit intelligent behavior
2. Learning can be implemented in hardware
3. Intelligence isn't exclusively biological

The principles influenced later work in AI, robotics, and reinforcement learning.

---

*Theseus proved that with the right circuits, machines could learn.*