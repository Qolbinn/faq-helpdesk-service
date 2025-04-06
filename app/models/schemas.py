from pydantic import BaseModel
from typing import List, Optional

class FAQItem(BaseModel):
    id: int
    question: str
    answer: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    threshold: Optional[float] = 0.6
    top_k: Optional[int] = 3

class QueryResult(BaseModel):
    id: int
    question: str
    answer: str
    similarity: float

class QueryResponse(BaseModel):
    results: List[QueryResult]
    best_match: Optional[QueryResult] = None