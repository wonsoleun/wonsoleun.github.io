---
title: "Formula correction, reference following, section boost"
order: 3
date: "2026-11-18"
description: "Human-in-the-loop formula correction, graph RAG step 1, and section boost"
---

## 이번 주차 요약

| # | 문제 | 해결 | 결과 |
|---|---|---|---|
| 1 | 자동 변환 수식이 틀리거나 통째로 빠짐 | 사람이 원본과 비교해 고친 LaTeX를 parse 단계에서 적용 | 의심 수식 225개 중 209개 수정 |
| 2 | 1위 조항이 "다른 조항에 따라 계산하라"고만 하고 식은 없음 | 1위 조항이 참조하는 조항을 한 단계 따라가 함께 표시 | 웹에서 확인함 |
| 3 | Pt.2에서 주조품·단조품처럼 다른 제품 Section으로 감 | 질문 단어가 Section 제목과 맞으면 가점 | Top-1 0.85 → 0.92 (61문항) |

<br>

---

## 1. 수식 하드 코딩


| 순서 | 할 일 | |
|---|---|---|
|[1]| 수식 목록 뽑기 | Docling JSON의 수식마다: PDF 쪽, 위치, 자동 LaTeX, 원본 이미지 |
|[2]| 눈으로 확인 | 원본 이미지와 KaTeX로 그린 결과를 나란히 보고 맞음/틀림 표시 |
|[3]| 틀린 것만 기록 | fixes/formulas.jsonl 에 "틀린 LaTeX → 맞는 LaTeX" |
|[4]| 01 다시 실행 | parse 단계에서 fixes 파일을 적용 (PDF 재변환 없음) |
|[5]| 02 다시 실행 | 본문이 바뀌었으니 인덱스 재생성 |


### 과정

- DNV Rule SHIP Part 2 Materials and welding_S.pdf: 수식 11개 (그중 번호만 인식된 것 10개)
- DNV Rule SHIP Part 3 Hull - 2026_S.pdf: 수식 61개
- DNV Rule SHIP Part 5 Ship type - 2026_S.pdf: 수식 153개

수식 225개 → data/processed/formula_review_suspicious.html 통해서 직접 하나하나 확인했다............



### 결과

**검수 결과**

의심 수식 225개 중 **209개(93%)가 실제로 틀려** 수정했다. 맞은 것은 16개뿐이었다.
<br> → 의심 표시(관계 기호 없음, 본체가 짧음, 괄호 짝, 반복)가 틀린 수식을 꽤 정확하게 골라냄

| 수정 유형 | Part 2 | Part 3 | Part 5 | 합계 | 내용 |
|---|:---:|:---:|:---:|:---:|---|
| 빠진 본체 입력 | 10 | 0 | 0 | 10 | 번호 `( 1 )`만 인식된 자리에 식 본체를 넣음 |
| 잘못 인식 정정 | 0 | 42 | 29 | 71 | 잘리거나 깨진 LaTeX를 원본대로 다시 씀 |
| 수식 아님 제거 | 1 | 12 | 115 | 128 | 수식이 아닌데 수식으로 인식된 것을 본문에서 뺌 |
| **합계** | 11 | 54 | 144 | **209** | |

<br>

**검수한 수식 유형**

1. 빠진 본체 (Part 2)
- Docling이 식 번호 `( 1 )`만 수식으로 인식하고 왼쪽의 본체는 수식으로도, 그림으로도, 글자로도 인식하지 않음
- 파서는 번호만 있는 항목을 버리게 되어 있어서 식이 통째로 사라짐
- e.g. Pt.2 Ch.1 Sec.2 [3.3]: 본문이 `Equation (1):`에서 끝나고 Ceq·Pcm·CET 식이 없었음. Part 2의 다른 조항들이 "식은 Ch.1 Sec.2 [3.3] 참조"라고 하는 바로 그 식들
-> 번호만 인식된 항목을 "본체가 빠진 식"의 표시로 보고, 그 자리에 본체를 넣도록 함

2. 수식 아님 (주로 Part 5)
- 128개 중 83개가 IGC Code 참조 문구 `(IGC Code Ch.x x.x)`가 수식으로 인식된 것 (Pt.5 Ch.7 액화가스 운반선)
- 15개는 Guidance note 끝 표시 `---e-n-d---o-f---`가 깨져서 기존 쓰레기 필터를 빠져나간 것
- 나머지는 `f o r`처럼 조건 문장의 일부가 수식으로 잘려 나온 것

-> 검색 순위는 그대로. 검색이 아니라 **답변에 맞는 수식이 들어가게 하는** 작업이구나...

<br>

---

## 2. 참조 조항 따라가기 

### 문제
- 질문: "How is the effective breadth of attached plating calculated?"
- 1위 Pt.5 Ch.3 Sec.2 [5.1]: "... calculated according to **Pt.3 Ch.3 Sec.7 [1.3.2]**" → 식이 없음
- 정답 [1.3.2]는 **2위**, 1·2위 점수 차 0.06
-> 검색이 못 찾은 게 아니라, 질문과 문장이 더 비슷한 안내 조항에 1위를 내준 것


### 해결
- 파서가 이미 저장해 둔 조항 참조(`refs`)를 이용해, 1위 조항이 가리키는 조항을 한 단계 따라감
- 조항 참조만, 코퍼스에서 찾은 것만, 최대 2개. 검색 순위는 바꾸지 않음
- 참조 기준은 테스트셋에서 참고해서 0.75로 함
- 웹: 답변 아래 "참조 조항" 블록 / LLaMA: `referenced by [1]`로 함께 입력

```text
top = search(query)[0]
for ref in top.refs:                       # 검색이 일치한 문단의 참조 먼저
    if ref is 조항 참조 and 코퍼스에 있음:
        show(article_of(ref), highlight=ref)
```

<br>

---

## 3. Section 제목 일치에 가점 주기

### 문제
- Pt.2 Ch.2는 Section이 제품별(rolled steel, steel forgings, steel castings, clad steel …)이고, 같은 번호 조항의 문장이 거의 같음
- "steel **castings**" 질문에 forgings 조항이 1위로 나옴

### 해결
- 질문 단어와 Section 제목의 일치도(idf 가중, 0~1)를 계산해 가장 잘 맞는 Section 하나에만 가점 주기
- `steel`처럼 모든 제목에 있는 단어는 무게가 거의 0, `castings`·`clad`는 큼
- 범위는 Pt.2 Ch.2·Ch.3, 일치도 0.55 이상일 때만

```text
match = Σ idf(맞은 제목 단어) / Σ idf(제목 단어)
dense  += 5.0 × match      # 재정렬 후보 선정
rerank += 3.0 × match      # 최종 순서
```

### 결과

| | Top-1 | Top-5 | Pt.2 |
|---|:---:|:---:|:---:|
| base + rerank | 53 / 61 | 61 | 13 / 19 |
| **base + rerank + 가점** | **56 / 61 (0.92)** | **61** | **16 / 19** |

