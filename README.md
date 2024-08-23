# tei-client
[![PyPI Version](https://img.shields.io/pypi/v/tei-client.svg)](https://pypi.org/project/tei-client)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/tei-client.svg)](https://pypi.org/project/tei-client)

Convenience Client for [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) with synchronous and asynchronous HTTP/gRPC support.

Implements the API defined in [TEI Swagger](https://huggingface.github.io/text-embeddings-inference/).

## Installation

You can easily install `tei-client` via pip:

```shell
pip install tei-client
```

### Grpc Support
If you want to use grpc, you need to install `tei-client` with grpc support:
```shell
pip install tei-client[grpc]
```

## Usage

## Creating a Client

### HTTP Example

To create an instance of the client, you can do the following:
```python
from tei_client import HTTPClient

url = 'http://localhost:8080'
client = HTTPClient(url)
```
<details>
    <summary>Example docker server</summary>

```shell
docker run -p 8080:80 -v ./tei_data:/data ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id sentence-transformers/all-MiniLM-L6-v2
```

</details>


### gRPC Example

Alternatively, you can use gRPC to connect to your server:
```python
import grpc
from tei_client import GrpcClient

channel = grpc.insecure_channel('localhost:8080')
client = GrpcClient(channel)
```
<details>
    <summary>Example docker server</summary>

```shell
docker run -p 8080:80 -v ./tei_data:/data ghcr.io/huggingface/text-embeddings-inference:cpu-latest-grpc --model-id sentence-transformers/all-MiniLM-L6-v2
```

</details>

## Embedding

To generate embeddings, you can use the following methods:

#### Single Embedding Generation

You can generate a single embedding using the `embed` method:
```python
result = client.embed("This is an example sentence")
print(result[0])
```

#### Batch Embedding Generation

To generate multiple embeddings in batch mode, use the `embed` method with a list of sentences:
```python
results = client.embed(["This is an example sentence", "This is another example sentence"])
for result in results:
    print(result)
```

#### Asynchronous Embedding Generation

For asynchronous embedding generation, you can use the `async_embed` method:
```python
result = await client.async_embed("This is an example sentence")
```

## Classification

To generate classification results for a given text, you can use the following methods:

#### Basic Classification

You can classify a single text using the `classify` method:
```python
result = client.classify("This is an example sentence")
print(result[0].scores)
```

#### NLI Style Classification

For Natural Language Inference (NLI) style classification, you can use the `classify` method with tuples as input. The first element of the tuple represents the premise and the second element represents the hypothesis.

```python
premise = "This is an example sentence"
hypothesis = "An example was given"

result = client.classify((premise, hypothesis))
print(result[0].scores)
```

#### Asynchronous and Batched Classification

The `classify` method also supports batched requests by passing a list of tuples or strings. For asynchronous classification, you can use the `async_classify` method.

```python
# Classify multiple texts in batch mode
results = client.classify(["This is an example sentence", "This is another example sentence"])
for result in results:
    print(result.scores)

# Asynchronous classification
result = await client.async_classify("This is an example sentence")
```

## Reranking

Reranking allows you to refine the order of search results based on additional information. This feature is supported by the `rerank` method.

#### Basic Reranking

You can use the `rerank` method to rerank search results with the following syntax:
```python
result = client.rerank(
    query="What is Deep Learning?",  # Search query
    texts=["Lore ipsum", "Deep Learning is ..."]  # List of text snippets
)
print(result)
```
#### Asynchronous Reranking

For asynchronous reranking, use the `async_rerank` method:
```python
result = await client.async_rerank(
    query="What is Deep Learning?",  # Search query
    texts=["Lore ipsum", "Deep Learning is ..."]  # List of text snippets
)
```