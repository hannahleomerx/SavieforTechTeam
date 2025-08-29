__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')

from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict
import random
import re
from langdetect import detect, LangDetectException
from langchain_core.documents import Document


SUPPORTED_LANGS = {"de", "en", "es", "pt", "pl"}

class FaqToolInput(BaseModel):
    query: str = Field(..., description="The user question")
    k: Optional[int] = Field(2, description="Number of top documents to retrieve")
    language: Optional[str] = Field(None, description="ISO code of language to filter, e.g. 'de', 'en'")
    category: Optional[str] = Field(None, description="Category tag to filter by")

class FaqTool(BaseTool):
    name: str = "faq_tool"
    description: str = (
        "Du bist der Support-Bot für Saventic Care. Beantworte Fragen anhand der FAQ-Dokumente "
        "in der gewählten Sprache und Kategorie."
    )
    args_schema: Type[BaseModel] = FaqToolInput

    def __init__(self, persist_directory: str, collection_name: str, prompt_template: str):
        super().__init__()
        self._llm             = ChatOpenAI(temperature=0.1)
        self._embeddings      = OpenAIEmbeddings(model="text-embedding-3-small")
        self._store           = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self._embeddings
        )
        self._prompt_template = prompt_template

    def _configure_chain(self, language: Optional[str], category: Optional[str], k: int) -> None:
        filters = {
            **({"language": language} if language else {}),
            **({"category": category} if category else {})
        }
        retriever = self._store.as_retriever(search_kwargs={
            "k": k,
            "filter": filters
        })
        suffix = f"(Kategorie: {category or 'alle'}, Sprache: {language or 'alle'})"
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self._prompt_template + "\n\n" + suffix
        )

        if k <= 1:
            chain_type   = "stuff"
            chain_kwargs = {"prompt": prompt}
        else:
            chain_type   = "refine"
            chain_kwargs = {
                "question_prompt": prompt,
                "refine_prompt": PromptTemplate(
                    input_variables=["existing_answer", "context", "question"],
                    template=(
                        "Bisherige Antwort:\n{existing_answer}\n\n"
                        "Weiterer Kontext:\n{context}\n\n"
                        "Bitte verfeinere die Antwort auf: {question}"
                    )
                ),
                "document_variable_name": "context"
            }

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_kwargs
        )
        # remenber for suggestions
        self._retriever = retriever

    def _run(
            self,
            query: str,
            k: int = 2,
            language: Optional[str] = None,
            category: Optional[str] = None
    ) -> Dict[str, Any]:
        # 0) Identify language, if not set
        if language is None:
            try:
                detected = detect(query)
            except LangDetectException:
                detected = None
            if detected in SUPPORTED_LANGS:
                language = detected

        # 1) Clean up query & format of question
        user_query = query.strip()
        if not user_query.endswith("?"):
            user_query += "?"
        qa_query = user_query
        if "saventic care" not in user_query.lower():
            qa_query = f"Saventic Care: {user_query}"

        # 2) Configure Q&A-Chain
        self._configure_chain(language, category, k)
        qa_res = self._qa_chain({"query": qa_query})
        raw_answer = qa_res["result"]
        answer = re.sub(r'(?m)^A:\s*', "", raw_answer).strip()
        sources = [doc.metadata for doc in qa_res["source_documents"]]

        # 3) get suggestion with k*5
        filters = {
            **({"language": language} if language else {}),
            **({"category": category} if category else {})
        }
        suggest_retriever = self._store.as_retriever(search_kwargs={
            "k": k * 5,
            "filter": filters
        })
        docs = suggest_retriever.get_relevant_documents(user_query)

        # 4) 3 suggestions based on paraphrases
        cand_questions = []
        for d in docs:
            p = d.metadata.get("paraphrase", "").strip()
            if not p:
                # Fallback auf erste Q:-Zeile
                for line in d.page_content.splitlines():
                    if line.startswith("Q:"):
                        p = line[len("Q:"):].strip()
                        break
            if p:
                if not p.endswith("?"):
                    p += "?"
                cand_questions.append(p)

        random.shuffle(cand_questions)
        suggestions = []
        seen = set()
        for q in cand_questions:
            if q.lower() == user_query.lower() or q in seen:
                continue
            seen.add(q)
            suggestions.append(q)
            if len(suggestions) >= 3:
                break

        return {
            "answer": answer,
            "sources": sources,
            "suggestions": suggestions
        }

    async def _arun(
            self,
            query: str,
            k: int = 2,
            language: Optional[str] = None,
            category: Optional[str] = None
    ) -> Dict[str, Any]:
        # 0) Identify language, if not set
        if language is None:
            try:
                detected = detect(query)
            except LangDetectException:
                detected = None
            if detected in SUPPORTED_LANGS:
                language = detected

        # 1) Clean up query & format of question
        user_query = query.strip()
        if not user_query.endswith("?"):
            user_query += "?"
        qa_query = user_query
        if "saventic care" not in user_query.lower():
            qa_query = f"Saventic Care: {user_query}"

        # 2) Configure Q&A-Chain
        self._configure_chain(language, category, k)
        qa_res = await self._qa_chain.arun({"query": qa_query})
        raw_answer = qa_res["result"]
        answer = re.sub(r'(?m)^A:\s*', "", raw_answer).strip()
        sources = [doc.metadata for doc in qa_res["source_documents"]]

        # 3) get suggestion with k*5
        filters = {
            **({"language": language} if language else {}),
            **({"category": category} if category else {})
        }
        suggest_retriever = self._store.as_retriever(search_kwargs={
            "k": k * 5,
            "filter": filters
        })
        docs = suggest_retriever.get_relevant_documents(user_query)

        # 4) suggestions based on paraphrases
        cand_questions = []
        for d in docs:
            p = d.metadata.get("paraphrase", "").strip()
            if not p:
                for line in d.page_content.splitlines():
                    if line.startswith("Q:"):
                        p = line[len("Q:"):].strip()
                        break
            if p:
                if not p.endswith("?"):
                    p += "?"
                cand_questions.append(p)

        random.shuffle(cand_questions)
        suggestions = []
        seen = set()
        for q in cand_questions:
            if q.lower() == user_query.lower() or q in seen:
                continue
            seen.add(q)
            suggestions.append(q)
            if len(suggestions) >= 3:
                break

        return {
            "answer": answer,
            "sources": sources,
            "suggestions": suggestions
        }








