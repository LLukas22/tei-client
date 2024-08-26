from src.tei_client import GrpcClient
from src.tei_client import ModelType, ClassificationTuple
import pytest


EMBED_URL = "localhost:8081"
CLASSIFIER_URL = "localhost:8083"
RERANKER_URL = "localhost:8085"


def test_is_constructable():
	client = GrpcClient(EMBED_URL)
	assert client.health()


@pytest.mark.parametrize(
	"url,model_type",
	[
		(EMBED_URL, ModelType.Embedding),
		(CLASSIFIER_URL, ModelType.Classifier),
		(RERANKER_URL, ModelType.Reranker),
	],
)
def test_info(url: str, model_type: ModelType):
	client = GrpcClient(url)
	info = client.info()
	assert info.server_model_type == model_type


@pytest.mark.parametrize(
	"url,model_type",
	[
		(EMBED_URL, ModelType.Embedding),
		(CLASSIFIER_URL, ModelType.Classifier),
		(RERANKER_URL, ModelType.Reranker),
	],
)
async def test_info_async(url: str, model_type: ModelType):
	client = GrpcClient(url)
	info = await client.async_info()
	assert info.server_model_type == model_type


def test_embed():
	client = GrpcClient(EMBED_URL)
	result = client.embed("Hello world")
	assert len(result) == 1

	result = client.embed(["Hello world", "This is a good day"])
	assert len(result) == 2


async def test_async_embed():
	client = GrpcClient(EMBED_URL)
	result = await client.async_embed("Hello world")
	assert len(result) == 1

	result = await client.async_embed(["Hello world", "This is a good day"])
	assert len(result) == 2


async def test_async_embed_bulk():
	client = GrpcClient(EMBED_URL)
	result = await client.async_embed(["Hello world"] * 512)
	assert len(result) == 512


def test_embed_all():
	client = GrpcClient(EMBED_URL)
	result = client.embed_all("Hello world")
	assert len(result) == 1
	assert len(result[0][0]) > 1


async def test_async_embed_all():
	client = GrpcClient(EMBED_URL)
	result = await client.async_embed_all("Hello world")
	assert len(result) == 1
	assert len(result[0][0]) > 1


def test_tokenize():
	client = GrpcClient(EMBED_URL)
	result = client.tokenize("Hello world")
	assert len(result) == 1

	result = client.tokenize(["Hello world", "foo bar"])
	assert len(result) == 2


async def test_async_tokenize():
	client = GrpcClient(EMBED_URL)
	result = await client.async_tokenize("Hello world")
	assert len(result) == 1

	result = await client.async_tokenize(["Hello world", "foo bar"])
	assert len(result) == 2


def test_decode():
	client = GrpcClient(EMBED_URL)
	result = client.tokenize("hello world")
	result = client.decode(result[0].get_ids())
	assert result == "hello world"


async def test_async_decode():
	client = GrpcClient(EMBED_URL)
	result = await client.async_tokenize("hello world")
	result = await client.async_decode(result[0].get_ids())
	assert result == "hello world"


@pytest.mark.parametrize(
	"inputs,expected_results",
	[
		("Hello World", 1),
		(["Hello world", "foo bar"], 2),
		(("Hello world", "foo bar"), 1),
		([("Hello world", "foo bar"), ("Hallo", "Hello")], 2),
		(ClassificationTuple("Hallo", "Hello"), 1),
		(
			[
				ClassificationTuple("Hallo", "Hello"),
				ClassificationTuple("Hallo2", "Hello2"),
			],
			2,
		),
	],
)
def test_classify(inputs, expected_results: int):
	client = GrpcClient(CLASSIFIER_URL)
	result = client.classify(inputs)
	assert len(result) == expected_results
