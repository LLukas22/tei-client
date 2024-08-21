Create grpc stubs: 

```
python -m grpc_tools.protoc -I.  --proto_path=./external/text-embeddings-inference/proto  --python_out=./src/tei_client/stubs --grpc_python_out=./src/tei_client/stubs --pyi_out=./src/tei_client/stubs tei.proto
```

Run GRPC server: 

```
docker run -p 8080:80 -v ./tei:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest-grpc --model-id BAAI/bge-m3
```