try:
	import grpc  # noqa: F401

	SUPPORTS_GRPC = True
except ImportError:
	SUPPORTS_GRPC = False


from tei_client.clients.http_client import HttpClient

if SUPPORTS_GRPC:
	from tei_client.clients.grpc_client import GrpcClient  # noqa: F401

from tei_client.models import (
	ModelType,
	ClassificationInput,
	EmbeddingInput,
	ClassificationTuple,
)

__all__ = [
	"HttpClient",
	"SUPPORTS_GRPC",
	"ModelType",
	"ClassificationTuple",
	"ClassificationInput",
	"EmbeddingInput",
]
