from logging import debug

from tei_client.clients.http import HttpClient

SUPPORTS_GRPC = False
try:
    import grpc
    from tei_client.clients.grpc import GrpcClient

    SUPPORTS_GRPC = True
except ImportError as e:
    debug(
        "Could not import 'grpc', if you want to use the GrpcClient install the package with grpc support via: `pip install tei-client[grpc]`"
    )
