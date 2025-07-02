from langchain.tools import BaseTool
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Type, Optional
from pydantic.v1 import Extra

class FaqToolInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="The user question"
    )

class FaqTool(BaseTool):
    name: str = "user_query"
    description: str = "Retrieve an answer from the FAQ database for questions regarding studying."
    args_schema: Type[BaseModel] = FaqToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, store: Chroma):
        """Initialisiere das Tool und lade den Index"""
        super().__init__()
        self.store = store

    def _run(self, query) -> object:
        try:
            results = self.store.similarity_search(
                query,
                k=2,
            )
            return results
        except Exception as e:
            return f"Error querying items: {str(e)}"
