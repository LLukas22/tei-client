from logging import debug

from tei_client.clients.http import HttpClient as HttpClient

SUPPORTS_GRPC = False
try:
	import grpc  # noqa: F401
	from tei_client.clients.grpc import GrpcClient  # noqa: F401

	SUPPORTS_GRPC = True
except ImportError:
	debug(
		"Could not import 'grpc', if you want to use the GrpcClient install the package with grpc support via: `pip install tei-client[grpc]`"
	)
