from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pydantic.v1 import Extra
from tantivy import Index, Occur, Query, Searcher
from typing import Type,  List
import re


class RecommenderToolInput(BaseModel):
    titles: List[str] = Field(..., description="Series titles for which to obtain recommendations")


class RecommenderTool(BaseTool):
    name: str = "series_recommendation_query"
    description: str = "Retrieve similar series with a more like this query"
    args_schema: Type[BaseModel] = RecommenderToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, index: Index, top_k: int = 5) -> object:
        """Initialisiere das Tool und lade den Index"""
        super().__init__()
        self.index = index
        self.top_k = top_k

    def _run(self, titles: List[str]) -> str:
        """Finde ähnliche TV-Serien auf Basis von Beschreibung und Genre"""
        searcher: Searcher = self.index.searcher()
        tmdb_path = "https://image.tmdb.org/t/p/w200"

        try:
            results = []
            for title in titles:
                query = self.index.parse_query(title, ['title'])
                hits = searcher.search(query, 1).hits
                results = []
                for (score, address) in hits:
                    hit = searcher.doc(address)
                    queries = []
                    description = hit["tmdb_overview"][0] + " " + hit["description"][0]
                    # Entferne leere Zeilen und Doppelpunkte
                    clean_description = "\n".join(line for line in description.splitlines() if line.strip()).replace(
                        ":", "")
                    # Finde alle Zeichen, die keine Buchstaben(a - z, A - Z), keine
                    # Ziffern(0 - 9) und keine Whitespace - Zeichen sind.
                    query_str = re.sub(r'[^a-zA-Z0-9\s]', '', clean_description)
                    # Entferne eckige Klammern
                    cleaned_text = re.sub(r'\[[^\]]*\]', '', query_str)
                    mlt_query = self.index.parse_query(cleaned_text, ['description'])
                    queries.append((Occur.Should, mlt_query))
                    genres = []
                    for i, genre in enumerate(hit["genres"]):
                        query = self.index.parse_query(f'{genre}', ["genres"])
                        queries.append((Occur.Should, query))
                        genres.append(genre)
                    for i, genre in enumerate(hit["tmdb_genre_ids"]):
                        query = self.index.parse_query(f'{genre}', ["tmdb_genre_ids"])
                        queries.append((Occur.Should, query))
                    boolean_query = Query.boolean_query(queries)
                    recommendations = searcher.search(boolean_query, limit=self.top_k).hits
                    for sim_score, doc_address in recommendations:
                        series = searcher.doc(doc_address)
                        json_obj = {
                            "title": series["title"][0],
                            "id": series["id"][0],
                            "url": series["url"][0],
                            "poster": tmdb_path+series["tmdb_poster_path"][0],
                            "description": series["tmdb_overview"][0]
                        }
                        results.append(json_obj)
            return results

        except Exception as e:
            return f"Error finding similar movies: {str(e)}"
