import grpc
import stubs.tei_pb2_grpc as tei_pb2_grpc
import stubs.tei_pb2 as tei_pb2

if __name__ == '__main__':
    with grpc.insecure_channel('localhost:8080') as channel:
        stub = tei_pb2_grpc.InfoStub(channel)
        result = stub.Info(tei_pb2.InfoRequest())
        print(result)
        
        