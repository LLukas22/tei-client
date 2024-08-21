Create grpc stubs: 

```
python -m grpc_tools.protoc -I.  --proto_path=./external/text-embeddings-inference/proto  --python_out=./src/tei_client/grpc --grpc_python_out=./src/tei_client/grpc tei.proto
```