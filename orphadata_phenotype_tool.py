# orphadata_phenotype_tool.py

import requests
from typing import List, Dict, Any
from pydantic import PrivateAttr
from langchain.tools import BaseTool

class OrphadataPhenotypeTool(BaseTool):
    name: str = "orphadata_phenotype_tool"
    description: str = (
        "Lädt die HPO-Phänotypen zu einer seltenen Erkrankung via ORPHAcode."
    )
    _base_url: str = PrivateAttr()

    def __init__(self, base_url: str = "https://api.orphadata.com"):
        super().__init__()
        self._base_url = base_url.rstrip("/")

    def _run(self, orpha_code: str, lang_code: str = "EN") -> List[Dict[str, Any]]:
        return self.get_phenotypes(orpha_code, lang_code)

    async def _arun(self, orpha_code: str, lang_code: str = "EN") -> List[Dict[str, Any]]:
        return self.get_phenotypes(orpha_code, lang_code)

    def get_phenotypes(self, orpha_code: str, lang_code: str = "EN") -> List[Dict[str, Any]]:
        endpoint = f"{self._base_url}/rd-phenotypes/orphacodes/{orpha_code}"
        resp = requests.get(
            endpoint,
            params={"language": lang_code},
            headers={"Accept": "application/json"},
            timeout=5
        )
        if resp.status_code != 200:
            return []

        try:
            payload = resp.json().get("data", {}).get("results", {})
        except ValueError:
            return []

        # hier nehmen wir den korrekten Pfad
        assoc = payload.get("Disorder", {}) \
                       .get("HPODisorderAssociation", [])

        phenotypes: List[Dict[str, Any]] = []
        for entry in assoc:
            hpo = entry.get("HPO", {})
            hid  = hpo.get("HPOId")
            name = hpo.get("HPOTerm")
            freq = entry.get("HPOFrequency")  # jetzt HPOFrequency statt Frequency
            if hid and name and freq:
                phenotypes.append({
                    "HPOId": hid,
                    "Name": name,
                    "HPOFrequency": freq
                })

        return phenotypes
