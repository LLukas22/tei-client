from tei_client.clients import HttpClient, SUPPORTS_GRPC

if SUPPORTS_GRPC:
	from tei_client.clients import GrpcClient as GrpcClient

from tei_client.models import ModelType, ClassificationInput, EmbeddingInput, ClassificationTuple

__all__ = ['HttpClient', 'SUPPORTS_GRPC', 'ModelType', 'ClassificationTuple', 'ClassificationInput', 'EmbeddingInput']