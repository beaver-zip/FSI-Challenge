# -*- coding: utf-8 -*-
"""
PDF 문서들을 읽어 Faiss 벡터 데이터베이스를 구축하고 저장합니다.
"""

import os
import glob
from tqdm import tqdm

# src 폴더에 있는 모듈을 사용하기 위해 경로 추가
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import DocumentProcessor # hanseo/RAG/Preprocessor/preprocessor.py -> src/preprocessor.py
from src.rag import LocalVectorDB # hanseo/RAG/VDB/localVDB.py -> src/rag.py

# --- 경로 설정 ---
PDF_PATH = "data/raw/RAG_pdfs"
FAISS_INDEX_PATH = "db/faiss_index.bin"
DOC_META_PATH = "db/doc_metadata.pkl"
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean"

def main():
    """
    FAISS 인덱스 파일을 생성합니다.
    """
    print(f"FAISS 인덱스 생성을 시작합니다. PDF 경로: {PDF_PATH}")

    # DocumentProcessor와 LocalVectorDB 인스턴스 생성
    doc_processor = DocumentProcessor()
    vdb = LocalVectorDB(model_name=EMBEDDING_MODEL_NAME)

    # PDF 파일 목록 가져오기
    pdf_files = glob.glob(os.path.join(PDF_PATH, "*.pdf"))
    if not pdf_files:
        print(f"[경고] {PDF_PATH} 경로에 PDF 파일이 없습니다. 벡터 DB를 생성할 수 없습니다.")
        return

    all_chunks = []
    print(f"{len(pdf_files)}개의 PDF 파일을 처리합니다...")
    for pdf_file in tqdm(pdf_files, desc="PDF 파일 처리 중"):
        try:
            # 각 PDF를 청크로 분할
            chunks = doc_processor.preprocess(file_path=pdf_file)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue
    
    # 청크가 하나 이상 있을 경우, 인덱스 생성 및 저장
    if all_chunks:
        print(f"총 {len(all_chunks)}개의 청크로 벡터 DB를 생성합니다.")
        vdb.create_and_save_index(all_chunks, FAISS_INDEX_PATH, DOC_META_PATH)
        print(f"성공적으로 벡터 DB를 생성하여 다음 파일들로 저장했습니다:")
        print(f"- 인덱스: {FAISS_INDEX_PATH}")
        print(f"- 메타데이터: {DOC_META_PATH}")
    else:
        print("처리할 문서 청크가 없습니다. PDF 파일 내용을 확인하세요.")

if __name__ == "__main__":
    main()
