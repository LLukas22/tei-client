from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, NamedTuple, Tuple
from enum import Enum


class ClassificationTuple(NamedTuple):
	premise: str
	hypothesis: str


EmbeddingInput = Union[Union[str, list[str]], Union[list[int], list[list[int]]]]

SingleClassificationInput = Union[str, Tuple[str, str], ClassificationTuple]
ClassificationInput = Union[SingleClassificationInput, list[SingleClassificationInput]]


class TruncationDirection(str, Enum):
	Left = "Left"
	Right = "Right"


class ModelType(str, Enum):
	Embedding = "embedding"
	Classifier = "classifier"
	Reranker = "reranker"


class EmbeddingMetadata(BaseModel):
	pooling: Literal["cls", "mean", "splade", "last_token"]


class ClassifierMetadata(BaseModel):
	id2label: dict[str, str]
	label2id: dict[str, int]


ModelMetadata = Union[EmbeddingMetadata, ClassifierMetadata]


def get_model_metadata_prototype(model_type: ModelType) -> type[ModelMetadata]:
	if model_type == ModelType.Embedding:
		return EmbeddingMetadata
	elif model_type == ModelType.Classifier:
		return ClassifierMetadata
	elif model_type == ModelType.Reranker:
		return ClassifierMetadata


class Info(BaseModel):
	version: str
	sha: str
	docker_label: str
	server_model_id: str = Field(..., alias="model_id")
	server_model_sha: Optional[str] = Field(None, alias="model_sha")
	server_model_dtype: str = Field(..., alias="model_dtype")
	server_model_type: ModelType = Field(..., alias="model_type")
	server_model_metadata: Optional[ModelMetadata] = Field(None, alias="model_metadata")
	max_concurrent_requests: int
	max_input_length: int
	max_batch_tokens: int
	max_batch_requests: int
	max_client_batch_size: int
	tokenization_workers: int


class Token(BaseModel):
	id: int
	text: str
	special: bool
	start: Optional[int] = None
	stop: Optional[int] = None


class TokenizationResult(BaseModel):
	tokens: list[Token]

	def get_ids(self) -> list[int]:
		return [token.id for token in self.tokens]


class ClassificationScore(BaseModel):
	score: float
	label: str


class ClassificationResult(BaseModel):
	scores: list[ClassificationScore]


class RerankScore(BaseModel):
	score: float
	index: int
	text: Optional[str] = None


class RerankResult(BaseModel):
	ranks: list[RerankScore]
