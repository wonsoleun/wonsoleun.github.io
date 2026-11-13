---
title: "SEA-YA: Domain Specific Chatbot Project"
date: "2026-11-12"
description: "Development log of problems found, root-cause analyses, and fixes for the DNV regulation RAG chatbot, in chronological order"
author: "Soleun Won"
tags:
  - "RAG"
  - "Programming"
  - "Python"
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

<br>

### 문제: 수식이 들어간 조항을 검색하면 조항과 변수 정의는 나오지만, 수식 자체만 빠져서 답변에 보이지 않음

<br>

### 문제 파악

의심 지점을 앞에서부터 하나씩 지워 나간 결과, PDF 추출 단계에서 생긴 문제라는 걸 알 수 있었다. <br>
1. 검색 확인: 정답 조항(5.1)을 제대로 가져왔다

 → 검색 문제 아님
<br>

2. chunking 확인: 변수 정의가 같은 chunk에 있다

 → 수식과 where 절이 갈라진 문제 아님
<br>

3. PDF 원본 확인: 수식을 드래그하면 선택이 안 된다

 → 수식에 텍스트 레이어가 없음. 확인해 보니 이미지가 아니라 **벡터 path(선·곡선 도형)**로 그려져 있음
<br>

4. chunks.jsonl 확인: 남은 조각이 $$ R \text{for condition} A . $$로 감싸져 있다

 → Docling이 formula로 잡고 LaTeX 변환도 했지만, 꼬리 텍스트만 변환함
<br>

5. 같은 수식 블록의 나머지 조각(WSD condition B, PSF A·B 식)이 5.3 본문(p.13, chunk 2460/2461)에 들어가 있다

 → 원래 5.1에 있어야 하는데, 페이지가 넘어가면서 다른 조항으로 들어감

<br>

**핵심 문제**
1. 수식 자체가 빠지는 것
2. 수식과 꼬리 텍스트가 같은 줄인지 판정하는 방법 = bbox 두 개가 같은 줄에 있는지 판단하기
3. 페이지가 넘어가면 수식 조각이 다른 조항으로 분리되는 것

<br>

### - 원인
Docling 처리 순서: Layout 검출 → **Layout 후처리** → 읽기 순서 → **수식 변환(CodeFormula)**
- Layout 검출: 페이지 이미지를 보고 수식 줄 전체를 formula로 잘 잡음
- Layout 후처리: 영역의 bbox를 그 안의 PDF 텍스트 조각에 맞춰 다시 그림
<br> → 벡터 수식 본체에는 텍스트가 없으니, 영역이 꼬리 텍스트("R for condition A") 크기로 줄어듦
<br> → 영역 안에 텍스트가 하나도 없으면 원래 영역을 유지함 (p.66은 이 경우라 영역이 정상)
- 수식 변환: 줄어든 영역만 잘라서 LaTeX로 바꿈 → 꼬리만 변환됨
- 여백 설정(expansion_factor)은 bbox 비례라 작은 영역을 넓혀도 본체에 닿지 않음 → 설정으로는 해결 불가

<br>

### - 선행 연구

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

**왜 이게 문제였던걸까?**

1. Docling은 수식의 BBox를 내부 텍스트 셀의 크기에 맞춰 줄인다. (`_adjust_cluster_bboxes`).
2. 근데 수식 자체에는 텍스트 셀이 없으므로 BBox가 꼬리 텍스트 영역만 남게 된 것.
3. `CodeFormula`가 전달받은 잘린 영역에는 꼬리 텍스트만 포함되어 있어, 최종적으로 `R \text{for condition} A`와 같은 결과가 생성.

<br>

**해결 구조** (Docling 앞 단계와 parse.py는 그대로, prepare_element 직전에만 보정)
1. 후보 모으기: 줄어든 bbox와 같은 줄의 잉크 덩어리, PDF 이미지 객체
2. 걸러내기: 큰 그래픽, 위아래 다른 줄 제외
3. 합치기: 후보 + 꼬리 텍스트를 하나의 영역으로
4. 자르고 변환: 넓힌 영역으로 잘라 기존 CodeFormula에 넣기

<br>

**결과** (Part 3의 p.12·13·66, 보정 끄고/켜고 비교)

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

## - bbox 같은 줄 판단

