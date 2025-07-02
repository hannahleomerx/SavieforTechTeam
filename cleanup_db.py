# cleanup_db.py
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

client = vector_store._client
collection = client.get_collection("example_collection")

# selektiv löschen, z.B. alle FAQ-Einträge
collection.delete(where={"Quelle": "FAQ"})
print("Alte FAQ-Einträge gelöscht.")
