from src.tei_client import HttpClient
from src.tei_client import ModelType
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
    assert info.model_type == model_type


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
    assert info.model_type == model_type


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
    