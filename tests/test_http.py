from tei_client import HttpClient
from tei_client import ModelType, ClassificationTuple
import pytest

EMBED_URL = "http://localhost:8080"
CLASSIFIER_URL = "http://localhost:8082"
RERANKER_URL = "http://localhost:8084"


def test_is_constructable():
	client = HttpClient(EMBED_URL)
	assert client.health()


async def test_is_constructable_async():
	client = HttpClient(EMBED_URL)
	assert await client.async_health()


@pytest.mark.parametrize(
	"url,model_type",
	[
		(EMBED_URL, ModelType.Embedding),
		(CLASSIFIER_URL, ModelType.Classifier),
		(RERANKER_URL, ModelType.Reranker),
	],
)
def test_info(url: str, model_type: ModelType):
	client = HttpClient(url)
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
	client = HttpClient(url)
	info = await client.async_info()
	assert info.server_model_type == model_type


def test_embed():
	client = HttpClient(EMBED_URL)
	result = client.embed("Hello world")
	assert len(result) == 1

	result = client.embed(["Hello world", "This is a good day"])
	assert len(result) == 2


async def test_async_embed():
	client = HttpClient(EMBED_URL)
	result = await client.async_embed("Hello world")
	assert len(result) == 1

	result = await client.async_embed(["Hello world", "This is a good day"])
	assert len(result) == 2


def test_embed_all():
	client = HttpClient(EMBED_URL)
	result = client.embed_all("Hello world")
	assert len(result) == 1
	assert len(result[0][0]) > 1


async def test_async_embed_all():
	client = HttpClient(EMBED_URL)
	result = await client.async_embed_all("Hello world")
	assert len(result) == 1
	assert len(result[0][0]) > 1


def test_tokenize():
	client = HttpClient(EMBED_URL)
	result = client.tokenize("Hello world")
	assert len(result) == 1

	result = client.tokenize(["Hello world", "foo bar"])
	assert len(result) == 2


async def test_async_tokenize():
	client = HttpClient(EMBED_URL)
	result = await client.async_tokenize("Hello world")
	assert len(result) == 1

	result = await client.async_tokenize(["Hello world", "foo bar"])
	assert len(result) == 2


def test_decode():
	client = HttpClient(EMBED_URL)
	result = client.tokenize("hello world")
	result = client.decode(result[0].get_ids())
	assert result[0] == "hello world"


async def test_async_decode():
	client = HttpClient(EMBED_URL)
	result = await client.async_tokenize(["hello world"])
	result = await client.async_decode(result[0].get_ids())
	assert len(result) == 1
	assert result[0] == "hello world"


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
	client = HttpClient(CLASSIFIER_URL)
	result = client.classify(inputs)
	assert len(result) == expected_results


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
async def test_async_classify(inputs, expected_results: int):
	client = HttpClient(CLASSIFIER_URL)
	result = await client.async_classify(inputs)
	assert len(result) == expected_results


def test_rerank():
	client = HttpClient(RERANKER_URL)
	result = client.rerank(
		query="What is Deep Learning?",
		texts=["Lore ipsum", "Deep Learning is ..."],
		return_text=True,
	)
	assert len(result.ranks) == 2
	assert result.ranks[0].index == 1
	assert result.ranks[0].text == "Deep Learning is ..."


async def test_async_rerank():
	client = HttpClient(RERANKER_URL)
	result = await client.async_rerank(
		query="What is Deep Learning?",
		texts=["Lore ipsum", "Deep Learning is ..."],
		return_text=True,
	)
	assert len(result.ranks) == 2
	assert result.ranks[0].index == 1
	assert result.ranks[0].text == "Deep Learning is ..."
