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
        """다양한 휴리스틱을 사용하여 문서의 제목을 추출합니다."""
        # 1. 메타데이터로 찾기
        title = doc.metadata.get('title', '')
        if title and isinstance(title, str) and len(title) > 4:
            cleaned_title = title.replace('.pdf', '').strip()
            if len(cleaned_title) > 4:
                return cleaned_title

        # 2. 첫 페이지에서 찾기
        if len(doc) > 0:
            page = doc[0]
            # A: 가장 큰 텍스트 찾기
            blocks = page.get_text("dict")["blocks"]
            max_font_size = 0
            largest_text = ""
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                largest_text = span["text"]
            
            largest_text = largest_text.strip()
            if len(largest_text) > 4:
                return largest_text

            # B: 일반적인 키워드가 포함된 텍스트 찾기
            lines = page.get_text().split('\n')
            candidates = []
            keywords = ['법', '시행령', '규정', 'Act', 'Law']
            for line in lines:
                if any(keyword in line for keyword in keywords):
                    cleaned_line = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', line).strip()
                    if 4 < len(cleaned_line) < 50: # 제목 길이 제한: 제목은 일반적으로 너무 길거나 짧지 않으므로
                        candidates.append(cleaned_line)
            if candidates:
                return min(candidates, key=len) # 가장 짧은 후보 반환

        # 3. 파일 이름으로 대체
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
