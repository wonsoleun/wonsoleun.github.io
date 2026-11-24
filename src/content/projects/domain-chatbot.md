---
title: "Domain Specific Chatbot Project"
description: "Development log of problems found, root-cause analyses, and fixes for the DNV regulation RAG chatbot, in chronological order"
tags:
  - "RAG"
  - "AX"
---

## Overview
A retrieval-augmented chatbot that answers questions about ship classification rules (DNV), built as part of a university research project.



### My Contributions

Analyzed formula extraction errors from PDF parsing and built a human-in-the-loop LaTeX correction workflow
Implemented cross-reference following to retrieve related rule clauses
Designed section-aware retrieval scoring on top of dense retrieval and cross-encoder reranking
Built a small evaluation set and measured retrieval accuracy at each stage



### Results

Most flagged formulas turned out to be real extraction errors; classified them into three error types and corrected them
Each retrieval stage (reranking, reference following, section boost) improved top-1 accuracy over the baseline



### Stack
Python, Docling, LLaMA 3.1 8B, sentence-transformers (bi-encoder, cross-encoder), KaTeX