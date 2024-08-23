from tei_client.clients import HttpClient as HttpClient, SUPPORTS_GRPC

if SUPPORTS_GRPC:
	from tei_client.clients import GrpcClient as GrpcClient

from tei_client.models import *
