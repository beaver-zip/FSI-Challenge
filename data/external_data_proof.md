# 외부 데이터 출처 정보

이 파일은 프로젝트에 사용된 외부 데이터의 출처를 증빙하기 위한 문서입니다.

## 1. 법률 문서 (DAPT 학습용)

- **데이터**: `data/raw/adapter_docs/` 폴더 내의 모든 `.doc` 파일
- **출처**: 국가법령정보센터 (https://www.law.go.kr)
- **수집 방법**: 각 법령 이름을 검색하여 '현행법령' 다운로드 기능을 통해 수집함.

## 2. 법률 문서 (RAG 구축용)

- **데이터**: `data/raw/RAG_pdfs/` 폴더 내의 모든 `.pdf` 파일
- **출처**: 국가법령정보센터 (https://www.law.go.kr)
- **수집 방법**: 각 법령 이름을 검색하여 '현행법령' 다운로드 기능을 통해 수집함.

## 3. MCQA 데이터셋

- **데이터**: `data/raw/mcqa_dataset/` 폴더 내 파일
- **출처**: Hugging Face Hub - `aiqwe/FinShibainu`
- **링크**: https://huggingface.co/datasets/aiqwe/FinShibainu
- **수집 방법**: `scripts/0_download_assets.py` 스크립트를 통해 다운로드함.
