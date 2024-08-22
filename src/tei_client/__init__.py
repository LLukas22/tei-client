from tei_client.clients import HttpClient, SUPPORTS_GRPC

if SUPPORTS_GRPC:
    from tei_client.clients import GrpcClient

from tei_client.models import *
