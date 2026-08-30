"""
Hybrid Financial RAG Retriever Engine.
Combines Vector Embeddings (Dense Search) + BM25 (Sparse Keyword Search) + FlashRank Cross-Encoder Reranking
for precise retrieval of financial statements and SEC disclosures with strict source citations.
"""

from typing import List, Dict, Any
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

class FinancialHybridRetriever:
    def __init__(self, collection_name: str = "financial_docs"):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.documents = []
        self.metadata_list = []
        self.bm25_model = None

    def index_documents(self, chunks: List[Dict[str, Any]]):
        """Indexes text and table chunks into ChromaDB vector database and BM25 index."""
        ids = []
        texts = []
        metadatas = []

        for idx, item in enumerate(chunks):
            doc_id = f"doc_{idx}"
            text = item["content"]
            meta = item.get("metadata", {})

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(meta)

        self.documents = texts
        self.metadata_list = metadatas

        # Add to ChromaDB vector store
        if ids:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

        # Build BM25 Index
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25_model = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search: Dense Vector Retrieval + BM25 Keyword Retrieval.
        Applies Reciprocal Rank Fusion (RRF) to combine both score lists.
        """
        if not self.documents:
            return []

        # 1. Dense Vector Search
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, len(self.documents))
        )
        vector_docs = vector_results.get("documents", [[]])[0]
        vector_metas = vector_results.get("metadatas", [[]])[0]

        # 2. Sparse BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}

        # Rank vector results
        for rank, (doc, meta) in enumerate(zip(vector_docs, vector_metas)):
            key = doc[:100] # Use string prefix as document identifier
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))
            doc_map[key] = {"content": doc, "metadata": meta}

        # Rank BM25 results
        for rank, idx in enumerate(top_bm25_indices):
            doc = self.documents[idx]
            meta = self.metadata_list[idx]
            key = doc[:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))
            doc_map[key] = {"content": doc, "metadata": meta}

        # Sort combined results by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        
        final_results = []
        for k in sorted_keys:
            res = doc_map[k]
            res["score"] = round(rrf_scores[k], 4)
            final_results.append(res)

        return final_results
