from abc import ABC, abstractmethod
from tei_client.models import Info, ModelType, TruncationDirection


class TEIClient(ABC):
    def __init__(self):
        self.__model_type = None

    @property
    def model_type(self) -> ModelType:
        if not self.__model_type:
            self.__model_type = self.info().model_type
        return self.__model_type
    
    def _ensure_model_type(self, wanted_type:ModelType):
        """
        Throws an error if the model type is not the one expected
        """
        assert (
            self.model_type == wanted_type
        ), f"{wanted_type} model required. The model on the server is of type {self.model_type}"

    @abstractmethod
    def health(self) -> bool:
        """
        Check if the TEI server is alive and ready to serve requests
        """

    @abstractmethod
    async def async_health(self) -> bool:
        """
        Check if the TEI server is alive and ready to serve requests
        """

    @abstractmethod
    def info(self) -> Info:
        """
        Get information about the loaded model of the TEI server
        """

    @abstractmethod
    async def async_info(self) -> Info:
        """
        Get information about the loaded model of the TEI server
        """

    @abstractmethod
    def embed(
        self,
        text: str | list[str],
        normalize: bool = True,
        truncate: bool = False,
        truncation_direction: TruncationDirection = TruncationDirection.Right,
    ) -> list[list[float]]:
        """
        Generate embeddings for the given text
        """
        
    @abstractmethod
    async def async_embed(
        self,
        text: str | list[str],
        normalize: bool = True,
        truncate: bool = False,
        truncation_direction: TruncationDirection = TruncationDirection.Right,
    ) -> list[list[float]]:
        """
        Generate embeddings for the given text
        """
