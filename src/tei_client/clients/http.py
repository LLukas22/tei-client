import httpx
from typing import Any
from tei_client.clients.base import (
	ModelTypeMixin,
	AsyncClientMixin,
	ConcurrentClientMixin,
)
from tei_client.models import (
	ClassificationInput,
	ClassificationResult,
	Info,
	ModelType,
	RerankResult,
	TokenizationResult,
	TruncationDirection,
	get_model_metadata_prototype,
	EmbeddingInput,
	Token,
	ClassificationScore,
	RerankScore,
)


class HttpClient(ConcurrentClientMixin, AsyncClientMixin, ModelTypeMixin):
	def __init__(self, url: str, **kwargs) -> None:
		self.client = httpx.Client(base_url=url, **kwargs)
		self.async_client = httpx.AsyncClient(base_url=url, **kwargs)
		super().__init__()

	def health(self) -> bool:
		result = self.client.get("/info")
		return not result.is_error

	async def async_health(self) -> bool:
		result = await self.async_client.get("/info")
		return not result.is_error

	@staticmethod
	def _into_info(json: dict[str, Any]) -> Info:
		model_type_field = json.pop("model_type")
		model_type = next(iter(model_type_field.keys()))
		metadata = model_type_field[model_type]

		model_type = ModelType(model_type)
		prototype = get_model_metadata_prototype(model_type)
		model_metadata = prototype.model_validate(metadata)

		return Info(model_type=model_type, model_metadata=model_metadata, **json)

	def info(self) -> Info:
		result = self.client.get("/info")
		return HttpClient._into_info(result.json())

	async def async_info(self) -> Info:
		result = await self.async_client.get("/info")
		return HttpClient._into_info(result.json())

	def embed(
		self,
		text: EmbeddingInput,
		pool: bool = True,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		result = self.client.post(
			"/embed",
			json={
				"inputs": text,
				"normalize": normalize,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)
		return result.json()

	async def async_embed(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		result = await self.async_client.post(
			"/embed",
			json={
				"inputs": text,
				"normalize": normalize,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)
		return result.json()

	def embed_all(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[list[float]]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		result = self.client.post(
			"/embed_all",
			json={
				"inputs": text,
				"normalize": normalize,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)
		return result.json()

	async def async_embed_all(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		result = await self.async_client.post(
			"/embed_all",
			json={
				"inputs": text,
				"normalize": normalize,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)
		return result.json()

	def tokenize(
		self, text: str | list[str], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		if isinstance(text, str):
			text = [text]

		result = self.client.post(
			"/tokenize", json={"inputs": text, "add_special_tokens": add_special_tokens}
		)
		results = result.json()
		return [
			TokenizationResult(tokens=[Token.model_validate(t) for t in r])
			for r in results
		]

	async def async_tokenize(
		self, text: str | list[str], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		if isinstance(text, str):
			text = [text]

		result = await self.async_client.post(
			"/tokenize", json={"inputs": text, "add_special_tokens": add_special_tokens}
		)
		results = result.json()
		return [
			TokenizationResult(tokens=[Token.model_validate(t) for t in r])
			for r in results
		]

	def decode(
		self,
		tokenized_input: list[int] | list[list[int]],
		skip_special_tokens: bool = True,
	) -> str:
		result = self.client.post(
			"/decode",
			json={"ids": tokenized_input, "skip_special_tokens": skip_special_tokens},
		)
		return result.json()

	async def async_decode(
		self,
		tokenized_input: list[int] | list[list[int]],
		skip_special_tokens: bool = True,
	) -> str:
		result = await self.async_client.post(
			"/decode",
			json={"ids": tokenized_input, "skip_special_tokens": skip_special_tokens},
		)
		return result.json()

	@staticmethod
	def _prepare_classify_input(inputs: ClassificationInput) -> ClassificationInput:
		if isinstance(inputs, str):
			return [[inputs]]
		elif isinstance(inputs, tuple):
			return [list(inputs)]
		elif isinstance(inputs, list) and not any(
			isinstance(el, list) or isinstance(el, tuple) for el in inputs
		):
			return [[i] for i in inputs]

		return inputs

	def classify(
		self,
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[ClassificationResult]:
		self._ensure_model_type(ModelType.Classifier)

		inputs = HttpClient._prepare_classify_input(inputs)

		result = self.client.post(
			"/predict",
			json={
				"inputs": inputs,
				"raw_scores": raw_scores,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)

		results = result.json()
		return [
			ClassificationResult(
				scores=[ClassificationScore.model_validate(s) for s in r]
			)
			for r in results
		]

	async def async_classify(
		self,
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[ClassificationResult]:
		self._ensure_model_type(ModelType.Classifier)

		inputs = HttpClient._prepare_classify_input(inputs)

		result = await self.async_client.post(
			"/predict",
			json={
				"inputs": inputs,
				"raw_scores": raw_scores,
				"truncate": truncate,
				"truncation_direction": truncation_direction.value,
			},
		)

		results = result.json()
		return [
			ClassificationResult(
				scores=[ClassificationScore.model_validate(s) for s in r]
			)
			for r in results
		]

	def rerank(
		self,
		query: str,
		texts: list[str],
		return_text: bool = False,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> RerankResult:
		self._ensure_model_type(ModelType.Reranker)

		result = self.client.post(
			"/rerank",
			json={
				"query": query,
				"texts": texts,
				"return_text": return_text,
				"raw_scores": raw_scores,
				"truncate": truncate,
				"truncation_direction": truncation_direction,
			},
		)
		results = result.json()
		return RerankResult(ranks=[RerankScore.model_validate(r) for r in results])

	async def async_rerank(
		self,
		query: str,
		texts: list[str],
		return_text: bool = False,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> RerankResult:
		self._ensure_model_type(ModelType.Reranker)

		result = await self.async_client.post(
			"/rerank",
			json={
				"query": query,
				"texts": texts,
				"return_text": return_text,
				"raw_scores": raw_scores,
				"truncate": truncate,
				"truncation_direction": truncation_direction,
			},
		)
		results = result.json()
		return RerankResult(ranks=[RerankScore.model_validate(r) for r in results])
