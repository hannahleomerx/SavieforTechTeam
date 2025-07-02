from dotenv import load_dotenv
import os, yaml
import re
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Environment & API Key ---
load_dotenv()

# --- Pfade & Parameter ---
FAQ_DIR     = "documents/FAQ"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION  = "example_collection"

# Mehrsprachigkeit
languages = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "pl": "Polish",
}

# --- LLM & Embeddings ---
llm        = ChatOpenAI(temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Chains definieren ---
translation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["text","target_language"],
        template="Übersetze ins {target_language}: {text}"
    )
)
paraphrase_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question","language"],
        template="""
Formuliere 3 alternative, gängige Fragestellungen auf {language} zu:
{question}
Gib jede Paraphrase in einer neuen Zeile aus.
"""
    )
)

# --- Einlesen & Paraphrasieren - Antworten übersetzen---
docs = []
for fname in os.listdir(FAQ_DIR):
    if not fname.endswith(".txt"):
        continue
    raw = open(os.path.join(FAQ_DIR, fname), encoding="utf-8", errors="ignore").read().strip()

    # Frontmatter (optional)
    metadata = {}
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        metadata.update(yaml.safe_load(fm))
        if "tags" in metadata and isinstance(metadata["tags"], list):
            metadata["tags"] = ", ".join(metadata["tags"])
    else:
        body = raw



    # --- Body säubern (entfernt führende Leerzeilen) ---
    body = body.lstrip()

    # --- In mehrere Q:-Segmente splitten ---
    # jedes Segment beginnt mit einer Zeile, die mit "Q:" anfängt
    segments = re.split(r'(?m)^(?=Q:)', body)

    for seg in segments:
        # entferne komplett leere Zeilen
        lines = [l for l in seg.splitlines() if l.strip()]
        if not lines or not lines[0].startswith("Q:"):
            continue

        # erste Zeile ist Q: Frage(n)
        raw_question = lines[0][2:].strip()
        # Rest ist die Antwort
        raw_answer = "\n".join(lines[1:]).strip()

        # Varianten zulassen, getrennt durch Slash
        raw_variants = [q.strip() for q in raw_question.split("/") if q.strip()]

        for base_question in raw_variants:
            # --- Hier kommt dein bestehender Übersetzungs-/Paraphrasen-Loop hin ---
            for code, name in languages.items():
                # Frage übersetzen (für Deutsch nur Original)
                q_translated = (
                    translation_chain.run(text=base_question, target_language=name).strip()
                    if code != "de" else base_question
                )
                # Antwort übersetzen (für Deutsch nur Original)
                a_translated = (
                    translation_chain.run(text=raw_answer, target_language=name).strip()
                    if code != "de" else raw_answer
                )
                # Paraphrasen erzeugen...
                para_text = paraphrase_chain.run(question=q_translated, language=name)
                variants = [q_translated] + [
                    p.strip("- ").strip() for p in para_text.splitlines() if p.strip()
                ]

                # letztlich Document-Objekte anlegen
                for variant in variants:
                    # Für Deutsch das Original, sonst die übersetzte Antwort verwenden
                    answer_text = raw_answer if code == "de" else a_translated
                    content = f"Q: {variant}\nA: {answer_text}"
                    doc_meta = {
                        "id": raw_question,
                        "paraphrase": variant,
                        "language": code,
                    }
                    doc_meta.update(metadata)
                    docs.append(Document(
                        page_content=content,
                        metadata=doc_meta,
                        id=str(uuid4())
                    ))

# --- Neuer Flat-Index via from_documents() ---
# 1) Verzeichnis ./chroma_langchain_db VORHER löschen!
store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    # Flat-Index aktivieren
    collection_metadata={"index_factory": "flat"}
)

print(f"Ingested {len(docs)} FAQ-Varianten in Collection '{COLLECTION}'.")