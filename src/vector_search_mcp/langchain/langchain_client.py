
import os

from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LangChainOpenAIClient(BaseModel):

    openai_key: str = Field(default=os.getenv("OPENAI_API_KEY",""), alias="openai_key")
    azure_openai: bool = Field(default=os.getenv("AZURE_OPENAI","false").lower() == "true", alias="azure_openai")
    azure_openai_api_version: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_API_VERSION",""), alias="azure_openai_api_version")
    azure_openai_endpoint: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_ENDPOINT",""), alias="azure_openai_endpoint")
    openai_base_url: Optional[str] = Field(default=os.getenv("OPENAI_BASE_URL",""), alias="openai_base_url")
    embedding_model: str = Field(default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), alias="default_embedding_model")

    def get_embedding_client(self):
        if not self.embedding_model:
            raise ValueError("embedding_model is not set.")

        params = self.create_client_params()
        params["model"] = self.embedding_model
        if (self.azure_openai):
            # modelを設定する。
            embeddings = AzureOpenAIEmbeddings(
                **params
            )
        else:
            embeddings = OpenAIEmbeddings(
                **params
            )
        return embeddings
        
    def create_client_params(self) -> dict:
        if self.azure_openai:
            return self.__create_azure_openai_dict()
        else:
            return self.__create_openai_dict()
        
    def __create_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        return completion_dict

    def __create_azure_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        else:
            completion_dict["azure_endpoint"] = self.azure_openai_endpoint
            completion_dict["api_version"] = self.azure_openai_api_version
        return completion_dict
    