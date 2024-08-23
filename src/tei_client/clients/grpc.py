import grpc
import tei_client.stubs.tei_pb2_grpc as tei_pb2_grpc
import tei_client.stubs.tei_pb2 as tei_pb2

from tei_client.clients.base import TEIClient
from tei_client.models import Info


class GrpcClient(TEIClient):
	def __init__(self, channel: grpc.Channel) -> None:
		self.info_stub = tei_pb2_grpc.InfoStub(channel)

	def info(self) -> Info:
		result = self.info_stub.Info(tei_pb2.InfoRequest())
		return Info(version=result.version)
