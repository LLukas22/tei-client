import httpx
from typing import Any
from tei_client.clients.base import TEIClient
from tei_client.models import (
    Info,
    ModelType,
    TruncationDirection,
    get_model_metadata_prototype,
)


class HttpClient(TEIClient):
    def __init__(self, url: str, **kwargs) -> None:
        self.client = httpx.Client(base_url=url, **kwargs)
        self.async_client = httpx.AsyncClient(base_url=url, **kwargs)
        super().__init__()

    def health(self) -> bool:
        result = self.client.get("/info")
        return not result.is_error

    async def async_health(self) -> bool:
        result = await self.async_client.get("/info")
        return not result.is_error

    @staticmethod
    def _into_info(json: dict[str, Any]) -> Info:
        model_type_field = json.pop("model_type")
        model_type = next(iter(model_type_field.keys()))
        metadata = model_type_field[model_type]

        model_type = ModelType(model_type)
        prototype = get_model_metadata_prototype(model_type)
        model_metadata = prototype.model_validate(metadata)

        return Info(model_type=model_type, model_metadata=model_metadata, **json)

    def info(self) -> Info:
        result = self.client.get("/info")
        return HttpClient._into_info(result.json())

    async def async_info(self) -> Info:
        result = await self.async_client.get("/info")
        return HttpClient._into_info(result.json())

    def embed(
        self,
        text: str | list[str],
        normalize: bool = True,
        truncate: bool = False,
        truncation_direction: TruncationDirection = TruncationDirection.Right,
    ) -> list[list[float]]:
        self._ensure_model_type(ModelType.Embedding)
        
        if isinstance(text, str):
            text = [text]

        result = self.client.post(
            "/embed",
            json={
                "inputs": text,
                "normalize": normalize,
                "truncate": truncate,
                "truncation_direction": truncation_direction.value,
            },
        )
        return result.json()
    
    async def async_embed(self, text: str | list[str], normalize: bool = True, truncate: bool = False, truncation_direction: TruncationDirection = TruncationDirection.Right) -> list[list[float]]:
        self._ensure_model_type(ModelType.Embedding)
        
        if isinstance(text, str):
            text = [text]

        result = await self.async_client.post(
            "/embed",
            json={
                "inputs": text,
                "normalize": normalize,
                "truncate": truncate,
                "truncation_direction": truncation_direction.value,
            },
        )
        return result.json()
