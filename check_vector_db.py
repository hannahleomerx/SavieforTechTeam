import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --- 1) Vector Store initialisieren ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# --- 2) Dateien im FAQ-Verzeichnis zählen ---
directory_path = 'documents/FAQ'
files = [
    f for f in os.listdir(directory_path)
    if os.path.isfile(os.path.join(directory_path, f))
]
print("Dateien im Verzeichnis:", len(files))

# --- 3) Collection laden und Einträge zählen ---
client = vector_store._client
collection = client.get_collection("example_collection")
count_in_db = collection.count()
print("Dokumente in der DB:", count_in_db)

# --- 4) Roh-Metadaten aller Einträge ausgeben ---
all_meta = collection.get(include=["metadatas"])["metadatas"]
print("\n=== Roh-Metadaten aller Einträge ===")
for i, m in enumerate(all_meta):
    print(f"{i:02d} → {m}")

# --- 5) Dateinamen sicher extrahieren ---
filenames_in_db = [
    m.get("Dateiname")
    or m.get("filename")
    or "<ohne-Dateiname-Metadatum>"
    for m in all_meta
]
print("\nDateien in DB (aus Metadaten):", filenames_in_db)

# --- 6) Fehlende Dateien ermitteln ---
existing = set(fn for fn in filenames_in_db if not fn.startswith("<"))
missing = set(files) - existing
if missing:
    print("\n❌ Fehlende Dateien:", missing)
else:
    print("\n✅ Alle Dateien wurden übertragen.")
