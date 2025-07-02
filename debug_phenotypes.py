import requests
import json

def main():
    url = "https://api.orphadata.com/rd-phenotypes/orphacodes/355"
    params = {"language": "EN"}
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR beim Request: {e}")
        return

    data = resp.json()

    # 1) Dump der kompletten JSON (pretty print)
    print("---- vollständige JSON-Antwort ----")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # 2) Zeige die Top-Level-Schlüssel
    print("\n---- Top-Level keys in data ----")
    for k in data.keys():
        print("•", k)

    # 3) Zeige keys unter data.data
    dd = data.get("data", {})
    print("\n---- Keys in data['data'] ----")
    for k in dd.keys():
        print("•", k)

    # 4) Zeige keys unter data.data.results
    res = dd.get("results", {})
    print("\n---- Keys in data['data']['results'] ----")
    if isinstance(res, dict):
        for k in res.keys():
            print("•", k)
    else:
        print("results ist keine dict, sondern:", type(res))

    # 5) Falls HPODisorderAssociation existiert, zeige die ersten paar Einträge
    phenos = res.get("HPODisorderAssociation")
    print("\n---- HPODisorderAssociation →", type(phenos))
    if phenos:
        for p in phenos[:5]:
            term = p.get("HPO", {}).get("HPOTerm", "<kein Term>")
            freq = p.get("Frequency", "<keine Frequency>")
            print(f"• {term:40} ➔ {freq}")
    else:
        print("Keine Einträge unter HPODisorderAssociation gefunden.")

if __name__ == "__main__":
    main()
