Create grpc stubs: 

```
python -m grpc_tools.protoc -I.  --proto_path=./external/text-embeddings-inference/proto  --python_out=./src/tei_client/stubs --grpc_python_out=./src/tei_client/stubs --pyi_out=./src/tei_client/stubs tei.proto
```

Run test servers: 

```
docker compose up -d
```