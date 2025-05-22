from pydantic import BaseModel


class DummyPredictResponse(BaseModel):
    id: int
    sugar_content: float
