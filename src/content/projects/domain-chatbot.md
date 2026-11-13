---
title: "SEA-YA: Domain Specific Chatbot Project"
description: "Development log of problems found, root-cause analyses, and fixes for the DNV regulation RAG chatbot, in chronological order"
tags:
  - "RAG"
  - "Programming"
  - "Python"
external_url: "https://github.com/wonsoleun/Domain-specific-ChatBot"
---

# Sea-Ya 


## -  Baseline

참고 논문: Han et al. (2026), "A RAG-based Q&A system for ship regulations applying domain adaptation

<br>

DNV PDF ─▶ Docling JSON(수식 LaTeX·표 구조) ─▶ ① 계층형 DB 구축 ─▶ ② base 인덱스 ─▶ ③ GPL 학습데이터 ─▶ ④ Domain adaptation
                                                                       │
질의 ─▶ adapted bi-encoder 검색(Top-k) ─▶ 참조 표(HTML) 첨부 ─▶ LLaMA 3.1 8B (4-bit) ─▶ 답변 + 출처

<br>

---

## - 앞으로 고민해야할 과정 (계속 추가 예정)

1. 수식 
2. 이미지 출력
3. bbox 같은 줄 판단
4. 후보 k개를 늘어놓고 사용자가 고르게 하지 말고, 정답 조항과 그 근거를 확정적으로 할 것

-> 해당 피드백 부분은 아직 파싱 단계 정확도가 좋지 않아서 보류

<br>

---

## - Docling - formula


### Docling에 대해 이해해보자

*Livathinos, N., Auer, C., Staar, P.W.J., et al. (2025). Advanced Layout Analysis Models for Docling. arXiv:2509.11720*
<br> 
→ Docling 후처리는 레이아웃 박스를 PDF 텍스트 셀에 맞게 조이도록 설계됨

<br>

**핵심 문제**
1. 수식 자체가 빠지는 것
2. 수식과 꼬리 텍스트가 같은 줄인지 판정하는 방법 = bbox 두 개가 같은 줄에 있는지 판단하기
3. 페이지가 넘어가면 수식 조각이 다른 조항으로 분리되는 것 

<br>

---

### - formula 문제 해결을 위한 선행 연구 

**참고 논문**

*Shah, A.K., Dey, A., Zanibbi, R. (2021). A Math Formula Extraction and Evaluation Framework for PDF Documents. ICDAR 2021, LNCS 12822, pp. 19–34.*
<br>

→ 설계 원리: 수식 영역 경계는 페이지 이미지(잉크)로 정하고, PDF 텍스트는 그 안에 배정한다

*Mali, P., Kukkadapu, P., Mahdavi, M., Zanibbi, R. (2020). ScanSSD: Scanning Single Shot Detector for Mathematical Formulas in PDF Document Images. arXiv:2003.08005*
<br> 

→ 구체적 기법: 검출 박스를 닿아 있는 잉크 연결 요소에 맞춰 넓히거나 줄인다

- 적용 방향: 수식 검출은 Docling에 맡기고, 줄어든 formula bbox와 같은 줄의 잉크 덩어리만 모아 영역을 넓힘

<br>

### 추가 설명

**요약**

- `get_bitmap_rects()`는 이미지 객체만 찾아내는 함수
-> 이미지는 없고 수식만 있는 페이지에서 `get_bitmap_rects()` 비어 있음. <br>
-> 수식은 보이는데, 이미지 객체는 하나도 없다 <br>
-> 수식은 이미지가 아니라 다른 방식(벡터 path)으로 그려져 있구나 <br>

- 수식을 찾는 다른 방법으로, `_ink_components` 사용하자.
-> 수식이 path이던 이미지이던 감지 가능하니까


1. 이미지 객체가 하나도 없음

| 쪽 | text | path | image |
|:---:|---:|---:|---:|
| p.12 | 2026 | 10 | 0 |
| p.13 | 1674 | 71 | 0 |
| p.66 | 1578 | 102 | 0 |

수식이 이미지로 삽입되었다면 `image` 객체가 존재해야 하지만, 세 페이지 모두 이미지 객체가 0개


2. 수식 자리를 글자 크기의 `path`가 채우고 있음 (p.12)

`for condition A.` 문장이 있는 줄을 살펴보면

| 객체 | x 좌표 | 크기 | 설명 |
|:---|:---:|:---:|:---|
| path | 258–267 | 8.9 × 6.7 | 수식 본체의 일부 |
| path | 267–270 | 3.2 × 3.9 | 수식 본체의 일부 |
| ... | ... | ... | (7개 더 존재) |
| path | 304–310 | 5.5 × 6.7 | 수식 본체의 일부 |
| text | 316–384 | - | `for condition A.` |

다른 수식 줄에서도 마찬가지였다.

<br>

**원인 분석**

1. Docling은 수식의 BBox를 내부 텍스트 셀의 크기에 맞춰 줄인다. (`_adjust_cluster_bboxes`).
2. 근데 수식 자체에는 텍스트 셀이 없으므로 BBox가 꼬리 텍스트 영역만 남게 된 것.
3. `CodeFormula`가 전달받은 잘린 영역에는 꼬리 텍스트만 포함되어 있어, 최종적으로 `R \text{for condition} A`와 같은 결과가 생성.

<br>

**보정 후 결과** (Part 3의 p.12·13·66, 보정 끄고/켜고 비교)

| 페이지 | 보정 전 | 보정 후 |
| --- | --- | --- |
| p.12 | R for condition A | W_stat ≤ η_1 R for condition A |
| p.13 | R for condition B | W_stat + W_dyn ≤ η_2 R for condition B |
| p.13 | \frac{1}{p} for condition A | γ_stat-1 W_stat + γ_dyn-1 W_dyn ≤ R/γ_R ... |
| p.13 | \frac{1}{2} for condition B | γ_stat·2 W_stat + ... ≤ R/γ_R ... |
| p.66 | 보정 전후 같음 (영역이 원래 맞음) | - |

→ 세 쪽에서 영역이 잘려 있던 수식 4개가 모두 수식 자체까지 포함된 채 변환됨. 위아래 줄이 합쳐진 경우는 없었음

<br>

### 참고 링크
https://github.com/docling-project/docling/blob/main/docling/utils/layout_postprocessor.py

https://docling-project.github.io/docling/_generated/examples/develop_formula_understanding/

https://github.com/docling-project/docling/blob/main/docling/models/base_model.py

<br>

---

### - 문단이 넘어가면 읽기 순서에서 쪼개지는 문제 해결을 위한 선행 연구 

- 이 문제에 대한 보고된 한계 
*Auer, C., Lysak, M., Staar, P.W.J., et al. (2024). Docling Technical Report. arXiv:2408.09869* 에 따르면,
<br> → 원인 근거: 레이아웃 예측을 PDF 텍스트 토큰과 교차시켜 요소를 만드는 구조
<br> → 페이지 경계 단서: 단 끝에서 넘어가는 문단이 읽기 순서에서 쪼개지고 표가 끼어들 수 있음 

<br>

**해결방향**

1. 페이지를 넘는 구조 분석 (학습 기반)
Microsoft 연구팀(arXiv:2503.15893)은 분석을 페이지 단위와 문서 단위 두 단계로 나눈다. 문서 단위에서는 목차 추출, 계층 목록 추출, 페이지를 넘는 표 묶기, 페이지를 넘는 문단 묶기를 처리한다. DOSA(AACL 2026)도 같은 문제를 다루며, 문서를 청크 단위로 나눠 멀티모달 모델이 요소 사이의 계층, 순서 관계를 예측한다.
<br> → 범용적이지만 관계 라벨이 있는 학습 데이터가 필요하다


2. 기존 파서 출력 뒤에 규칙으로 보정하는 방향 (사후 보정)
Docling은 그대로 두고, 출력된 결과를 문서 단위로 다시 이어 붙인다. MinerU-Popo(arXiv:2605.24973)처럼 파서를 바꾸지 않고 뒤에서 고치는 방식이다.
<br> → DNV는 조항 번호 체계(Pt > Ch > Sec > [x.y.z])가 명확해서 학습 없이 규칙으로 판단할 근거가 많다
<br> → e.g. 페이지 맨 위에서 첫 조항 제목이 나오기 전의 요소는 이전 페이지 마지막 조항에 이어 붙인다


- 범용성은 1번이 더 좋지만, DNV처럼 계층이 확실한 문서는 2번으로도 충분할 것으로 판단됌 => 2번으로 결정

---

## - bbox 같은 줄 판단

