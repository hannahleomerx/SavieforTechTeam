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

# Muli languages
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

# --- Define Chains ---
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

# --- Fill and paraphrase - translate answers---
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



    # --- Clean up Body (deletes leading blank lines) ---
    body = body.lstrip()

    # --- Splitting in several Q segments ---
    # every segement starts with a line starting with Q:
    segments = re.split(r'(?m)^(?=Q:)', body)

    for seg in segments:
        # delete completly blank lines
        lines = [l for l in seg.splitlines() if l.strip()]
        if not lines or not lines[0].startswith("Q:"):
            continue

        # first line is Q: Question
        raw_question = lines[0][2:].strip()
        # rest is the answer
        raw_answer = "\n".join(lines[1:]).strip()

        # allow variants, split up by slash
        raw_variants = [q.strip() for q in raw_question.split("/") if q.strip()]

        for base_question in raw_variants:
            # --- translation paraphrase loop ---
            for code, name in languages.items():
                # Translate question (german only for orginal)
                q_translated = (
                    translation_chain.run(text=base_question, target_language=name).strip()
                    if code != "de" else base_question
                )
                # Translate answer (german only for original)
                a_translated = (
                    translation_chain.run(text=raw_answer, target_language=name).strip()
                    if code != "de" else raw_answer
                )
                # create paraphrases
                para_text = paraphrase_chain.run(question=q_translated, language=name)
                variants = [q_translated] + [
                    p.strip("- ").strip() for p in para_text.splitlines() if p.strip()
                ]

                # create document files
                for variant in variants:
                    # For german the original, other languages translated
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

# ---new flat index via from_documents() ---
# 1) Delete Repo ./chroma_langchain_db before
store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    # Activate flat-index
    collection_metadata={"index_factory": "flat"}
)


print(f"Ingested {len(docs)} FAQ-Varianten in Collection '{COLLECTION}'.")
