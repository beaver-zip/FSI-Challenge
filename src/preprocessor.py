import os
import re
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )

    def get_document_title(self, doc, file_path): # 문서 제목 추출
        """PDF 파일명에서 확장자를 제외한 부분을 제목으로 추출합니다."""
        base_name = os.path.basename(file_path)
        file_title = os.path.splitext(base_name)[0]
        # 파일 이름 정리
        file_title = re.sub(r'_\d+', '', file_title).replace('_', ' ').strip()
        return file_title

    def preprocess(self, file_path: str):
        """PDF 파일을 로드, 처리 및 청크로 분할합니다."""
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            return []

        legal_name = self.get_document_title(doc, file_path)
        print(f"Processing '{os.path.basename(file_path)}' with extracted title: \"{legal_name}\"\n")

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        doc.close()

        if not full_text.strip():
            return []

        text_chunks = self.text_splitter.split_text(full_text)

        processed_chunks = []
        for chunk in text_chunks:
            chunk_with_title = f'[{legal_name}] {chunk.strip()}'
            processed_chunks.append(chunk_with_title)
            
        return processed_chunks
