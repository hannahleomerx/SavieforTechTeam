# debug_retrieval.py
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

query = "Was ist Saventic Care?"
k = 10
docs = vector_store.similarity_search(query, k=k)

print(f"🔍 Top-{k} Chunks für '{query}':")
for i, d in enumerate(docs, 1):
    snippet = d.page_content.replace("\n", " ")[:300]
    print(f"{i}. «{snippet}…» — {d.metadata}")
