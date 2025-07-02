# wikidata_tool.py

from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Dict, Any, List
from langchain.tools import BaseTool
import requests
import urllib.parse
import streamlit as st

# Mapping von Session-Sprache zu Wikidata-Language-Code
LANG_MAP = {
    "de": "de",
    "en": "en",
    "pl": "pl",
    "es": "es",
    "pt": "pt"
}

class WikidataInput(BaseModel):
    name: str = Field(..., description="Krankheitsname (deutsch, englisch, polnisch, spanisch, portugiesisch)")
    info_type: str = Field("symptoms", description="Angefragte Info: 'symptoms', 'icd', 'related' usw.")

class WikidataTool(BaseTool):
    name: str = "wikidata_tool"
    description: str = (
        "Findet weiterführende Informationen zu Krankheiten über Wikidata. "
        "Gibt Symptome, ICD-10-Codes oder verwandte Krankheiten zurück."
    )
    args_schema: Type[WikidataInput] = WikidataInput

    _wikidata_url: str = PrivateAttr("https://www.wikidata.org/w/api.php")
    _sparql_url: str = PrivateAttr("https://query.wikidata.org/sparql")

    def _run(self, name: str, info_type: str = "symptoms") -> str:
        lang_code = LANG_MAP.get(st.session_state.lang, "en")
        qid = self._find_qid(name, lang_code)
        if not qid:
            return self._no_info_message(info_type, lang_code)

        if info_type == "symptoms":
            return self._fetch_symptoms(qid, lang_code)
        elif info_type == "icd":
            return self._fetch_icd(qid)
        elif info_type == "related":
            return self._fetch_related(qid, lang_code)
        else:
            return self._no_info_message(info_type, lang_code)

    async def _arun(self, name: str, info_type: str = "symptoms") -> str:
        return self._run(name, info_type)

    def _find_qid(self, disease_name: str, lang_code: str) -> str:
        """Findet die Wikidata Q-ID für eine Erkrankung anhand des Namens in der gewählten Sprache."""
        params = {
            "action": "wbsearchentities",
            "search": disease_name,
            "language": lang_code,
            "type": "item",
            "format": "json"
        }
        try:
            resp = requests.get(self._wikidata_url, params=params, timeout=5)
            results = resp.json().get("search", [])
            if results:
                return results[0]["id"]
        except Exception:
            pass
        return None

    def _fetch_symptoms(self, qid: str, lang_code: str) -> str:
        """Liest Symptome zu einer Q-ID aus Wikidata via SPARQL (mehrsprachig)."""
        query = f"""
        SELECT ?symptomLabel WHERE {{
          wd:{qid} wdt:P780 ?symptom.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang_code},en". }}
        }}
        """
        results = self._run_sparql(query)
        if not results:
            return self._no_info_message("symptoms", lang_code)
        symptoms = [r["symptomLabel"]["value"] for r in results]
        return self._format_multilang_response("symptoms", symptoms, lang_code)

    def _fetch_icd(self, qid: str) -> str:
        """Liest ICD-10-Codes aus Wikidata."""
        query = f"""
        SELECT ?code WHERE {{
          wd:{qid} wdt:P494 ?code.
        }}
        """
        results = self._run_sparql(query)
        if not results:
            return "No ICD-10 code found in Wikidata."
        codes = [r["code"]["value"] for r in results]
        return f"ICD-10-Code: {', '.join(codes)}"

    def _fetch_related(self, qid: str, lang_code: str) -> str:
        """Liest verwandte Krankheiten (mehrsprachig)."""
        query = f"""
        SELECT ?relLabel WHERE {{
          {{ wd:{qid} wdt:P279 ?rel. }} UNION
          {{ wd:{qid} wdt:P1542 ?rel. }} UNION
          {{ wd:{qid} wdt:P828 ?rel. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang_code},en". }}
        }}
        """
        results = self._run_sparql(query)
        if not results:
            return self._no_info_message("related", lang_code)
        rels = [r["relLabel"]["value"] for r in results]
        return self._format_multilang_response("related", rels, lang_code)

    def _run_sparql(self, query: str) -> List[Dict[str, Any]]:
        headers = {"Accept": "application/sparql-results+json"}
        try:
            resp = requests.get(self._sparql_url, params={"query": query}, headers=headers, timeout=10)
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except Exception:
            return []

    def _format_multilang_response(self, info_type: str, items: List[str], lang_code: str) -> str:
        info_titles = {
            "de": {"symptoms": "Symptome laut Wikidata", "related": "Verwandte Krankheiten laut Wikidata"},
            "en": {"symptoms": "Symptoms (Wikidata)", "related": "Related diseases (Wikidata)"},
            "pl": {"symptoms": "Objawy według Wikidata", "related": "Powiązane choroby według Wikidata"},
            "es": {"symptoms": "Síntomas según Wikidata", "related": "Enfermedades relacionadas según Wikidata"},
            "pt": {"symptoms": "Sintomas segundo o Wikidata", "related": "Doenças relacionadas segundo o Wikidata"}
        }
        title = info_titles.get(lang_code, info_titles["en"]).get(info_type, info_type.capitalize())
        return f"{title}: {', '.join(items)}"

    def _no_info_message(self, info_type: str, lang_code: str) -> str:
        messages = {
            "de": {"symptoms": "Keine Symptome in Wikidata hinterlegt.",
                   "icd": "Kein ICD-10-Code in Wikidata hinterlegt.",
                   "related": "Keine verwandten Krankheiten in Wikidata hinterlegt."},
            "en": {"symptoms": "No symptoms found in Wikidata.",
                   "icd": "No ICD-10 code found in Wikidata.",
                   "related": "No related diseases found in Wikidata."},
            "pl": {"symptoms": "Brak objawów w Wikidata.",
                   "icd": "Brak kodu ICD-10 w Wikidata.",
                   "related": "Brak powiązanych chorób w Wikidata."},
            "es": {"symptoms": "No se encontraron síntomas en Wikidata.",
                   "icd": "No se encontró el código ICD-10 en Wikidata.",
                   "related": "No se encontraron enfermedades relacionadas en Wikidata."},
            "pt": {"symptoms": "Nenhum sintoma encontrado no Wikidata.",
                   "icd": "Nenhum código ICD-10 encontrado no Wikidata.",
                   "related": "Nenhuma doença relacionada encontrada no Wikidata."}
        }
        return messages.get(lang_code, messages["en"]).get(info_type, "No information found.")

