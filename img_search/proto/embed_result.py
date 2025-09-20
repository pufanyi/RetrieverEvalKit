import numpy as np
from pydantic import BaseModel


class EmbedResult(BaseModel):
    model_name: str
    dataset_name: str
    embedding: np.ndarray
