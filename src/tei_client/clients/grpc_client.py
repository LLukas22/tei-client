import grpc
from typing import Optional, Union

from tei_client.clients.base import (
	ConcurrentClientMixin,
	AsyncClientMixin,
	ModelTypeMixin,
)
from tei_client.models import (
	Info,
	ModelType,
	TruncationDirection,
	EmbeddingInput,
	Token,
	TokenizationResult,
	ClassificationInput,
	ClassificationResult,
	ClassificationScore,
	ClassificationTuple,
)

import tei_client.stubs.tei_pb2_grpc as tei_pb2_grpc
import tei_client.stubs.tei_pb2 as tei_pb2


def to_modeltype(grpc_modeltype: tei_pb2.ModelType) -> ModelType:
	if grpc_modeltype == tei_pb2.ModelType.MODEL_TYPE_EMBEDDING:
		return ModelType.Embedding
	elif grpc_modeltype == tei_pb2.ModelType.MODEL_TYPE_CLASSIFIER:
		return ModelType.Classifier
	elif grpc_modeltype == tei_pb2.ModelType.MODEL_TYPE_RERANKER:
		return ModelType.Reranker


def to_grpc_truncation(truncation: TruncationDirection) -> tei_pb2.TruncationDirection:
	if truncation == TruncationDirection.Left:
		return tei_pb2.TruncationDirection.TRUNCATION_DIRECTION_LEFT
	elif truncation == TruncationDirection.Right:
		return tei_pb2.TruncationDirection.TRUNCATION_DIRECTION_RIGHT


class Stubs:
	def __init__(self, channel: Union[grpc.Channel, grpc.aio.Channel]) -> None:
		self.info = tei_pb2_grpc.InfoStub(channel)
		self.tokenize = tei_pb2_grpc.TokenizeStub(channel)
		self.embed = tei_pb2_grpc.EmbedStub(channel)
		self.predict = tei_pb2_grpc.PredictStub(channel)
		self.rerank = tei_pb2_grpc.RerankStub(channel)


class GrpcClient(ModelTypeMixin):
	def __init__(
		self,
		target: str,
		credentials: Optional[grpc.ChannelCredentials] = None,
		**kwargs,
	) -> None:
		if credentials is not None:
			self.channel = grpc.secure_channel(target, credentials, **kwargs)
			self.async_channel = grpc.aio.secure_channel(target, credentials, **kwargs)
		else:
			self.channel = grpc.insecure_channel(target, **kwargs)
			self.async_channel = grpc.aio.insecure_channel(target, **kwargs)

		self._stubs = Stubs(channel=self.channel)
		self._async_stubs = Stubs(channel=self.async_channel)
		super().__init__()

	def __del__(self):
		self.channel.close()
		self.async_channel.close()

	def health(self) -> bool:
		state = self.async_channel.get_state()
		return (
			state == grpc.ChannelConnectivity.READY
			or state == grpc.ChannelConnectivity.IDLE
		)

	@staticmethod
	def _into_info(result) -> Info:
		return Info(
			version=result.version,
			sha=result.sha,
			docker_label=result.docker_label,
			model_id=result.model_id,
			model_sha=result.model_sha,
			model_dtype=result.model_dtype,
			model_type=to_modeltype(result.model_type),
			model_metadata=None,  # Not implemented for grpc
			max_concurrent_requests=result.max_concurrent_requests,
			max_input_length=result.max_input_length,
			max_batch_tokens=result.max_batch_tokens,
			max_batch_requests=result.max_batch_requests,
			max_client_batch_size=result.max_client_batch_size,
			tokenization_workers=result.tokenization_workers,
		)

	def info(self) -> Info:
		result = self._stubs.info.Info(tei_pb2.InfoRequest())
		return GrpcClient._into_info(result)

	async def async_info(self) -> Info:
		result = await self._async_stubs.info.Info(tei_pb2.InfoRequest())
		return GrpcClient._into_info(result)

	def embed(
		self,
		text: EmbeddingInput,
		normalize: bool = True,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		requests = [
			tei_pb2.EmbedRequest(
				inputs=t,
				truncate=truncate,
				normalize=normalize,
				truncation_direction=to_grpc_truncation(truncation_direction),
			)
			for t in text
		]

		return [r.embeddings for r in self._stubs.embed.EmbedStream(iter(requests))]

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

		async def gen():
			for t in text:
				yield tei_pb2.EmbedRequest(
					inputs=t,
					truncate=truncate,
					normalize=normalize,
					truncation_direction=to_grpc_truncation(truncation_direction),
				)

		call = self._async_stubs.embed.EmbedStream(gen())

		responses = []
		for i in range(len(text)):
			response = await call.read()
			responses.append(response.embeddings)

		return responses

	def embed_all(
		self,
		text: EmbeddingInput,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[list[float]]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		requests = [
			tei_pb2.EmbedAllRequest(
				inputs=t,
				truncate=truncate,
				truncation_direction=to_grpc_truncation(truncation_direction),
			)
			for t in text
		]

		return [
			[t.embeddings for t in r.token_embeddings]
			for r in self._stubs.embed.EmbedAllStream(iter(requests))
		]

	async def async_embed_all(
		self,
		text: EmbeddingInput,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[list[float]]:
		self._ensure_model_type(ModelType.Embedding)

		if isinstance(text, str):
			text = [text]

		async def gen():
			for t in text:
				yield tei_pb2.EmbedAllRequest(
					inputs=t,
					truncate=truncate,
					truncation_direction=to_grpc_truncation(truncation_direction),
				)

		call = self._async_stubs.embed.EmbedAllStream(gen())

		responses = []
		for i in range(len(text)):
			response = await call.read()
			responses.append([t.embeddings for t in response.token_embeddings])

		return responses

	def tokenize(
		self, text: str | list[str], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		if isinstance(text, str):
			text = [text]

		requests = [
			tei_pb2.EncodeRequest(
				inputs=t, add_special_tokens=add_special_tokens, prompt_name=None
			)
			for t in text
		]
		results = []
		for r in self._stubs.tokenize.TokenizeStream(iter(requests)):
			results.append(
				TokenizationResult(
					tokens=[
						Token(
							id=t.id,
							text=t.text,
							special=t.special,
							start=t.start,
							stop=t.stop,
						)
						for t in r.tokens
					]
				)
			)
		return results

	async def async_tokenize(
		self, text: str | list[str], add_special_tokens: bool = True
	) -> list[TokenizationResult]:
		if isinstance(text, str):
			text = [text]

		async def gen():
			for t in text:
				yield tei_pb2.EncodeRequest(
					inputs=t, add_special_tokens=add_special_tokens
				)

		call = self._async_stubs.tokenize.TokenizeStream(gen())
		results = []
		for i in range(len(text)):
			response = await call.read()
			results.append(
				TokenizationResult(
					tokens=[
						Token(
							id=t.id,
							text=t.text,
							special=t.special,
							start=t.start,
							stop=t.stop,
						)
						for t in response.tokens
					]
				)
			)
		return results

	def decode(
		self,
		tokenized_input: list[int] | list[list[int]],
		skip_special_tokens: bool = True,
	) -> Union[str, list[str]]:
		if isinstance(tokenized_input[0], int):
			requests = [
				tei_pb2.DecodeRequest(
					ids=tokenized_input, skip_special_tokens=skip_special_tokens
				)
			]
		else:
			requests = [
				tei_pb2.DecodeRequest(ids=ti, skip_special_tokens=skip_special_tokens)
				for ti in tokenized_input
			]

		results = [r.text for r in self._stubs.tokenize.DecodeStream(iter(requests))]
		if len(results) == 1:
			return results[0]
		return results

	async def async_decode(
		self,
		tokenized_input: list[int] | list[list[int]],
		skip_special_tokens: bool = True,
	) -> str:
		if isinstance(tokenized_input[0], int):
			requests = [
				tei_pb2.DecodeRequest(
					ids=tokenized_input, skip_special_tokens=skip_special_tokens
				)
			]
		else:
			requests = [
				tei_pb2.DecodeRequest(ids=ti, skip_special_tokens=skip_special_tokens)
				for ti in tokenized_input
			]

		call = self._async_stubs.tokenize.DecodeStream(iter(requests))
		results = []
		for i in range(len(requests)):
			response = await call.read()
			results.append(response.text)

		if len(results) == 1:
			return results[0]
		return results

	@staticmethod
	def _prepare_classify_input(
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> tuple[bool, list[Union[tei_pb2.PredictRequest, tei_pb2.PredictPairRequest]]]:
		if not isinstance(inputs, list):
			inputs = [inputs]

		requests = []
		is_pair = False
		for _input in inputs:
			if isinstance(_input, ClassificationTuple):
				requests.append(
					tei_pb2.PredictPairRequest(
						inputs=[_input.premise, _input.hypothesis],
						truncate=truncate,
						raw_scores=raw_scores,
						truncation_direction=to_grpc_truncation(truncation_direction),
					)
				)
				is_pair = True
			elif isinstance(_input, tuple):
				requests.append(
					tei_pb2.PredictPairRequest(
						inputs=[_input[0], _input[1]],
						truncate=truncate,
						raw_scores=raw_scores,
						truncation_direction=to_grpc_truncation(truncation_direction),
					)
				)
				is_pair = True
			else:
				requests.append(
					tei_pb2.PredictRequest(
						inputs=_input,
						truncate=truncate,
						raw_scores=raw_scores,
						truncation_direction=to_grpc_truncation(truncation_direction),
					)
				)

		return is_pair, requests

	def classify(
		self,
		inputs: ClassificationInput,
		raw_scores: bool = False,
		truncate: bool = False,
		truncation_direction: TruncationDirection = TruncationDirection.Right,
	) -> list[ClassificationResult]:
		self._ensure_model_type(ModelType.Classifier)

		is_pair, requests = GrpcClient._prepare_classify_input(
			inputs, raw_scores, truncate, truncation_direction
		)

		stream = (
			self._stubs.predict.PredictPairStream(iter(requests))
			if is_pair
			else self._stubs.predict.PredictStream(iter(requests))
		)

		results = []
		for response in stream:
			results.append(
				ClassificationResult(
					scores=[
						ClassificationScore(score=p.score, label=p.label)
						for p in response.predictions
					]
				)
			)
		return results
