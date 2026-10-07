# This code is not shared by Author, of "Production RAG with LangChain & Vector Databases – Full Course"
# of freecodecamp.org, but is based on the course content.
# I have taken screenshot and tried to get as much as possible.
# Video Link:- https://www.youtube.com/watch?v=mHxLXzYjQRE&t=26506s
# Timeline : 2:10
# RRF Score = 1 / (rank + k) where k is a constant, rank is the rank of the document in the results.

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from typing import List

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def test_query(query, name, retriever):
    """Test a query and show results"""
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"  {i+1}. {preview}")
    return results


# Test queries designed to challenge vector search
test_queries = [
    "SKU-7742X specifications",  # Exact product code
    "E_CONN_REFUSED error",  # Error code
    "How do I authenticate?",  # Semantic question
    "WCAG compliance",  # Acronym
    "router configuration",  # General semantic
]

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

print(f"Loaded {len(documents)} documents for hybrid search.")

vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name="hybrid_test"
)

# Create vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Return top 3

print("Vector retriever ready")

# BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(documents, k=3)  # Return top 3

print("BM25 retriever ready")

# Combine with EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # Equal weight to both
)

print("Hybrid retriever ready")

for query in test_queries:
    print("=" * 60)

    # Vector only
    vector_results = test_query(query, "VECTOR", vector_retriever)

    # BM25 only
    bm25_results = test_query(query, "BM25", bm25_retriever)

    # Hybrid
    hybrid_results = test_query(query, "HYBRID", ensemble_retriever)


# This below functio does same manually what is done by EnsembleRetriever internally.
# Hybrid retriever using weighted Reciprocal Rank Fusion (RRF)
# this is what esambleretriever did internally.
# combine multiple retrievers using weighted Reciprocal Rank Fusion (RRF).
def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""
    doc_scores = {}  # page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]


class HybridRetriever:
    """Production hybrid retriever with BM25 + Vector search"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create vector store and retriever
        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query: str) -> List[Document]:
        """Run hybrid search using weighted RRF"""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, documents: List[Document]):
        """Add new documents to both retrievers"""
        # Add to vector store
        self.vectorstore.add_documents(documents)

        # Recreate BM25 (it doesn't support incremental adds)
        # In Retrieval-Augmented Generation (RAG) systems, incremental updates refer to the ability to add,
        # modify, or delete documents in the retrieval database in real time (or dynamically) without
        # rebuilding the entire search index from scratch.
        all_docs = self.vectorstore.get()
        # recreating again, because BM25Retriever doesn't support incremental adds
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc) for doc in all_docs["documents"]], k=self.k
        )


# Usage
retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-7742X specifications")

for doc in results:
    print(doc.page_content[:100])
