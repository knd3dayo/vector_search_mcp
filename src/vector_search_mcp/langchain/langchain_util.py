
import json, sys
from typing import Any, Generator
from langchain.docstore.document import Document

import os
from typing import Any, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from vector_search_mcp.langchain.langchain_vector_db import LangChainVectorDB
from vector_search_mcp.langchain.vector_search_request import VectorSearchRequest
from vector_search_mcp.langchain.embedding_data import EmbeddingData
from vector_search_mcp.langchain.langchain_vector_db_chroma import LangChainVectorDBChroma
from vector_search_mcp.langchain.langchain_vector_db_pgvector import LangChainVectorDBPGVector

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class VectorDBItemBase(BaseModel):

    # コレクションの指定がない場合はデフォルトのコレクション名を使用
    DEFAULT_COLLECTION_NAME: ClassVar[str] = "ai_app_default_collection"
    FOLDER_CATALOG_COLLECTION_NAME: ClassVar[str] = "ai_app_folder_catalog_collection"

    name: str = Field(default="default")
    description: str = Field(default="Application default vector db")
    vector_db_type: int = Field(default=1, ge=1, le=3, description="1: Chroma, 2: PGVector, 3: Other")
    vector_db_url: str = Field(default=os.path.join(os.getenv("APP_DATA_PATH", ""), "server", "vector_db", "default_vector_db"))
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME)
    doc_store_url: str = Field(default=f'sqlite:///{os.path.join(os.getenv("APP_DATA_PATH", ""), "server", "vector_db", "default_doc_store.db")}')
    chunk_size: int = Field(default=4096)
    is_use_multi_vector_retriever: bool = Field(default=True)


    @field_validator("is_use_multi_vector_retriever")
    @classmethod
    def parse_bool_multi_vector(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.upper() == "TRUE"
        return False
        
    def get_vector_db_type_string(self) -> str:
        '''
        vector_db_typeを文字列で返す
        '''
        if self.vector_db_type == 0:
            return "Chroma"
        elif self.vector_db_type == 1:
            return "PGVector"
        elif self.vector_db_type == 2:
            return "Other"
        else:
            return "Unknown"

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
    
class LangChainChatParameter:
    def __init__(self, chat_request_dict: dict):

        # messagesを取得
        messages = chat_request_dict.get("messages", [])
        # messagesのlengthが0の場合はエラーを返す
        if len(messages) == 0:
            self.prompt = ""
        else:
            # messagesの最後のメッセージを取得
            last_message: dict = messages[-1]
            # contentを取得
            content = last_message.get("content", {})
            # contentのうちtype: textのもののtextを取得
            prompt_array = [ c["text"] for c in content if c["type"] == "text"]
            # prpmpt_arrayのlengthが0の場合はエラーを返す
            if len(prompt_array) > 0:
                # promptを取得
                self.prompt = prompt_array[0]
                # messagesから最後のメッセージを削除
                messages.pop()
            else:
                raise ValueError("prompt is empty")

        # messagesをjson文字列に変換
        chat_history_json = json.dumps(messages, ensure_ascii=False, indent=4)
        self.chat_history = LangChainChatParameter.convert_to_langchain_chat_history(chat_history_json)
        # デバッグ出力
        logger.debug ("LangChainChatParameter, __init__")
        logger.debug(f'prompt: {self.prompt}')
        logger.debug(f'chat_history: {self.chat_history}')

    @classmethod
    def convert_to_langchain_chat_history(cls, chat_history_json: str):
        # openaiのchat_historyをlangchainのchat_historyに変換
        langchain_chat_history : list[Any]= []
        chat_history = json.loads(chat_history_json)
        for chat in chat_history:
            role = chat["role"]
            content = chat["content"]
            if role == "system":
                langchain_chat_history.append(SystemMessage(content))
            elif role == "user":
                langchain_chat_history.append(HumanMessage(content))
            elif role == "assistant":
                langchain_chat_history.append(AIMessage(content))
        return langchain_chat_history


class LangChainUtil:


    @classmethod
    async def update_embeddings(cls, client: LangChainOpenAIClient, vectordb: VectorDBItemBase, embedding_data: EmbeddingData) -> dict:
        """
        ベクトルDBのコンテンツインデックスを更新する
        :param openai_props: OpenAIProps
        :param embedding_data: EmbeddingData
        :return: dict
        """

        
        # LangChainVectorDBを生成
        vector_db: LangChainVectorDB = LangChainUtil.get_vector_db(client, vectordb)
        await vector_db.update_embeddings(embedding_data)

        return {}   

    @classmethod
    def get_vector_db(cls, client: LangChainOpenAIClient, vectordb: VectorDBItemBase) -> LangChainVectorDB:

        vector_db_url = vectordb.vector_db_url
        if vectordb.is_use_multi_vector_retriever:
            doc_store_url = vectordb.doc_store_url
        else:
            doc_store_url = ""
        collection_name = vectordb.collection_name
        chunk_size = vectordb.chunk_size

        # ベクトルDBのタイプがChromaの場合
        if vectordb.vector_db_type == 1:
            return LangChainVectorDBChroma(
                langchain_openai_client = client,
                vector_db_url = vector_db_url,
                collection_name = collection_name,
                doc_store_url= doc_store_url, 
                chunk_size = chunk_size)
        
        # ベクトルDBのタイプがPostgresの場合
        elif vectordb.vector_db_type == 2:
            return LangChainVectorDBPGVector(
                langchain_openai_client = client,
                vector_db_url = vector_db_url,
                collection_name = collection_name,
                doc_store_url= doc_store_url, 
                chunk_size = chunk_size)
                
        else:
            # それ以外の場合は例外
            raise ValueError("VectorDBType is invalid")

    # ベクトル検索を行う
    @classmethod
    async def vector_search(cls, client: LangChainOpenAIClient, vectordb: VectorDBItemBase, vector_search_request: VectorSearchRequest) -> list[Document]:

        result_documents = []

        # debug request.nameが設定されているか確認
        if not vector_search_request.name:
            raise ValueError("request.name is not set")
        if not vector_search_request.query:
            raise ValueError("request.query is not set")


        langchain_db = LangChainUtil.get_vector_db(client, vectordb)

        # デバッグ出力
        logger.info('ベクトルDBの設定')
        logger.info(f'''
            name:{vectordb.name} vector_db_description:{vectordb.description} 
            VectorDBTypeString:{vectordb.get_vector_db_type_string()} VectorDBURL:{vectordb.vector_db_url} 
            CollectionName:{vectordb.collection_name}
            ChunkSize:{vectordb.chunk_size} IsUseMultiVectorRetriever:{vectordb.is_use_multi_vector_retriever}
            ''')


        logger.info(f'Query: {vector_search_request.query}')
        logger.info(f'SearchKwargs:{vector_search_request.search_kwargs}')
        documents =  await langchain_db.vector_search(vector_search_request.query, vector_search_request.search_kwargs)
        result_documents.extend(documents)

        return result_documents
    
