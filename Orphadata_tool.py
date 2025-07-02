# Orphadata_tool.py

from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, List, Dict, Any
from langchain.tools import BaseTool
import requests
import urllib.parse
import streamlit as st  # Um auf st.session_state.lang zuzugreifen

class RareDiseaseInput(BaseModel):
    name: str = Field(..., description="Der (Teil-)Name einer seltenen Erkrankung")

class RareDiseaseTool(BaseTool):
    name: str = "rare_disease_tool"
    description: str = (
        "Suche nach einer seltenen Erkrankung per (Teil-)Namen. "
        "Gibt Definition, ORPHAcode, Synonyme und Link zurück "
        "(Labels immer auf Englisch, Übersetzung im Agent)."
    )
    args_schema: Type[RareDiseaseInput] = RareDiseaseInput

    _base_url: str = PrivateAttr()

    def __init__(self, base_url: str = "https://api.orphadata.com"):
        super().__init__()
        # Entferne abschließenden Slash, falls vorhanden
        self._base_url = base_url.rstrip("/")

    def _run(self, name: str) -> str:
        # Name URL-kodieren und Sprache aus Session-State
        encoded_name = urllib.parse.quote(name)
        lang_code = st.session_state.lang.upper()

        # Endpoint zusammenbauen
        endpoint = f"{self._base_url}/rd-cross-referencing/orphacodes/names/{encoded_name}"

        # HTTP-Request mit Timeout- und Connection-Error-Behandlung
        try:
            resp = requests.get(
                endpoint,
                params={"language": lang_code},
                headers={"Accept": "application/json"},
                timeout=5
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return self._service_unavailable_message()

        # HTTP-Fehler behandeln
        status = resp.status_code
        if status == 404:
            return self._no_info_message()
        if 400 <= status < 500:
            return self._no_info_message()
        if 500 <= status < 600:
            return self._service_unavailable_message()

        # JSON parsen
        try:
            json_data = resp.json()
        except ValueError:
            return self._service_unavailable_message()

        # Ergebnisse extrahieren
        raw_results = json_data.get("data", {}).get("results")
        if raw_results is None:
            return self._no_info_message()

        # In Liste normalisieren
        if isinstance(raw_results, dict):
            data_list = [raw_results]
        elif isinstance(raw_results, list):
            data_list = raw_results
        else:
            return self._no_info_message()

        if not data_list:
            return self._no_info_message()

        first = data_list[0]
        # Versuch, das Ergebnis zu formatieren
        try:
            result_str = self._format_result(first)
        except Exception:
            return self._service_unavailable_message()

        # Bei Erfolg nur das formatierte Ergebnis zurückgeben
        return result_str

    async def _arun(self, name: str) -> str:
        return self._run(name)

    def _format_result(self, record: Dict[str, Any]) -> str:
        """
        Antwort immer mit englischen Labels – Übersetzung im Agent!
        """
        preferred = record.get("Preferred term", "-")
        orpha_code = record.get("ORPHAcode", "-")
        orpha_url = record.get("OrphanetURL", "")
        synonyms = record.get("Synonym", [])
        summary_info = record.get("SummaryInformation", [])
        definition = summary_info[0].get("Definition", "") if summary_info else ""

        return (
            f"**{preferred}** (ORPHAcode: {orpha_code})\n"
            f"Definition: {definition}\n"
            f"Synonyms: {', '.join(synonyms)}\n"
            f"More info: {orpha_url}"
        )

    def _no_info_message(self) -> str:
        """Englische Standardmeldung für 'keine Information gefunden'."""
        return "No information found for that disease."

    def _service_unavailable_message(self) -> str:
        """Englische Standardmeldung für 'Service nicht verfügbar'."""
        return "The service is currently unavailable. Please try again later."
