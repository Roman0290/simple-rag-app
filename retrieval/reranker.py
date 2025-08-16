from sentence_transformers import CrossEncoder

class Reranker:
    """Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2"""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k: int = None):
        """Rerank documents based on query and return sorted docs (optionally top_k)"""
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, score in scored_docs]
        if top_k:
            reranked = reranked[:top_k]
        return reranked
