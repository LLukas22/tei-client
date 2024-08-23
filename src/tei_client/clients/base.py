from abc import ABC, abstractmethod
from typing import Optional, Union
from tei_client.models import (
	Info,
	ModelType,
	TruncationDirection,
	EmbeddingInput,
	TokenizationResult,
	ClassificationInput,
	ClassificationResult,
	RerankResult,
)


class ZeroShotMixin(ABC):
	def zero_shot_classification(
		texts: str | list[str], labels: list[str]
	) -> list[str]:
		pass


class ModelTypeMixin(ABC):
	__model_type: Optional[ModelType] = None

	@property
	def model_type(self) -> ModelType:
		if not self.__model_type:
			self.__model_type = self.info().model_type
		return self.__model_type

	def _ensure_model_type(self, wanted_type: ModelType):
		"""
		Throws an error if the model type is not the one expected
		"""
		assert (
			self.model_type == wanted_type
		), f"{wanted_type} model required. The model on the server is of type {self.model_type}"


class ConcurrentClientMixin(ABC):
	@abstractmethod
	def health(self) -> bool:
		"""
		Check if the TEI server is alive and ready to serve requests
		"""

	@abstractmethod
	def info(self) -> Info:
		"""
		Get information about the loaded model of the TEI server
		"""

	@abstractmethod
	def embed(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		"""
		Generate embeddings for the given text
		"""

	@abstractmethod
	def embed_all(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[list[float]]]:
		"""
		Generate embeddings without application of pooling
		"""

	@abstractmethod
	def tokenize(
		self, text: Union[str, list[str]], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		"""
		Tokenize the given input
		"""

	@abstractmethod
	def decode(
		self,
		tokenized_input: Union[list[int], list[list[int]]],
		skip_special_tokens: bool = True,
	) -> str:
		"""
		Decode the given tokenized input
		"""

	@abstractmethod
	def classify(
		self,
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[ClassificationResult]:
		"""
		Classify the given inputs
		"""

	@abstractmethod
	def rerank(
		self,
		query: str,
		texts: list[str],
		return_text: bool = False,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> RerankResult:
		"""
		Get the reranked results for the given query and texts
		"""


class AsyncClientMixin(ABC):
	@abstractmethod
	async def async_health(self) -> bool:
		"""
		Check if the TEI server is alive and ready to serve requests
		"""

	@abstractmethod
	async def async_info(self) -> Info:
		"""
		Get information about the loaded model of the TEI server
		"""

	@abstractmethod
	async def async_embed(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		"""
		Generate embeddings for the given text
		"""

	@abstractmethod
	def async_embed_all(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[list[float]]]:
		"""
		Generate embeddings without application of pooling
		"""

	@abstractmethod
	def async_tokenize(
		self, text: Union[str, list[str]], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		"""
		Tokenize the given input
		"""

	@abstractmethod
	def async_decode(
		self,
		tokenized_input: Union[list[int], list[list[int]]],
		skip_special_tokens: bool = True,
	) -> str:
		"""
		Decode the given tokenized input
		"""

	@abstractmethod
	async def async_classify(
		self,
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[ClassificationResult]:
		"""
		Classify the given inputs
		"""

	@abstractmethod
	async def async_rerank(
		self,
		query: str,
		texts: list[str],
		return_text: bool = False,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> RerankResult:
		"""
		Get the reranked results for the given query and texts
		"""
