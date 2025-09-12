
import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer

class LocalVectorDB:
    def __init__(self, model_name='upskyy/bge-m3-korean', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.documents = []

    def create_and_save_index(self, docs, index_path, doc_path):
        print("문서 임베딩을 시작합니다...")
        self.documents = docs
        embeddings = self.model.encode(self.documents, convert_to_tensor=True, show_progress_bar=True)
        
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"FAISS 인덱스를 생성합니다. 벡터 차원: {embeddings_np.shape[1]}")
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        
        print(f"FAISS 인덱스를 '{index_path}'에 저장합니다.")
        faiss.write_index(self.index, index_path)
        
        print(f"문서 내용을 '{doc_path}'에 저장합니다.")
        with open(doc_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def load_index(self, index_path, doc_path):
        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            print("저장된 인덱스 또는 문서 파일이 없습니다. 새로 생성해야 합니다.")
            return False
            
        print(f"FAISS 인덱스를 '{index_path}'에서 로드합니다.")
        self.index = faiss.read_index(index_path)
        
        print(f"문서 내용을 '{doc_path}'에서 로드합니다.")
        with open(doc_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print("로드 완료.")
        return True

    def search(self, query, k=5):
        if self.index is None:
            raise Exception("인덱스가 로드되지 않았습니다. load_index()를 먼저 호출하세요.")
        
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.index.search(query_embedding, k)
        
        results = [self.documents[i] for i in indices[0]]
        return results
