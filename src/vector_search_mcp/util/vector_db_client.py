
from langchain.docstore.document import Document

from pydantic import BaseModel, Field

from vector_search_mcp.langchain.langchain_vector_db import LangChainVectorDB
from vector_search_mcp.model.models import EmbeddingData
from vector_search_mcp.langchain.langchain_vector_db_chroma import LangChainVectorDBChroma
from vector_search_mcp.langchain.langchain_vector_db_pgvector import LangChainVectorDBPGVector
from vector_search_mcp.langchain.langchain_client import LangChainOpenAIClient
from vector_search_mcp.model.models import VectorDBItemBase, VectorSearchRequest

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class VectorDBClient(BaseModel):

    # langchain OpenAI client
    langchain_openai_client: LangChainOpenAIClient = Field(default=LangChainOpenAIClient())
    # vector dbのリスト
    vector_dbs: list[VectorDBItemBase] = [VectorDBItemBase()]

    async def update_embeddings(self, embedding_data: EmbeddingData) -> dict:
        """
        ベクトルDBのコンテンツインデックスを更新する
        :param openai_props: OpenAIProps
        :param embedding_data: EmbeddingData
        :return: dict
        """
        # embedding_dataのvector_db_nameが設定されている場合は、そのvector_db_nameに対応するVectorDBItemBaseを取得
        vector_db_name = embedding_data.vector_db_name if embedding_data.vector_db_name else "default"
        vectordb = next((db for db in self.vector_dbs if db.name == vector_db_name), None)
        if not vectordb:
            raise ValueError(f"VectorDBItemBase with name {vector_db_name} not found")

        # LangChainVectorDBを生成
        langchain_vector_db = self.get_vector_db(vectordb.name)
        await langchain_vector_db.update_embeddings(embedding_data)

        return {}   

    def get_vector_db(self, vector_db_name: str) -> LangChainVectorDB:

        vectordb = next((db for db in self.vector_dbs if db.name == vector_db_name), None)
        if not vectordb:
            raise ValueError(f"VectorDBItemBase with name {vector_db_name} not found")

        vector_db_url = vectordb.vector_db_url
        collection_name = vectordb.collection_name
        chunk_size = vectordb.chunk_size

        # ベクトルDBのタイプがChromaの場合
        if vectordb.vector_db_type == 1:
            return LangChainVectorDBChroma(
                langchain_openai_client = self.langchain_openai_client,
                vector_db_url = vector_db_url,
                collection_name = collection_name,
                chunk_size = chunk_size)
        
        # ベクトルDBのタイプがPostgresの場合
        elif vectordb.vector_db_type == 2:
            return LangChainVectorDBPGVector(
                langchain_openai_client = self.langchain_openai_client,
                vector_db_url = vector_db_url,
                collection_name = collection_name,
                chunk_size = chunk_size)
                
        else:
            # それ以外の場合は例外
            raise ValueError("VectorDBType is invalid")

    # ベクトル検索を行う
    async def vector_search(self, vector_search_request: VectorSearchRequest) -> list[Document]:

        result_documents = []

        # debug request.vector_db_nameが設定されているか確認
        if not vector_search_request.vector_db_name:
            raise ValueError("request.vector_db_name is not set")
        if not vector_search_request.query:
            raise ValueError("request.query is not set")

        # vector_search_requestのvector_db_nameが設定されている場合は、そのvector_db_nameに対応するVectorDBItemBaseを取得
        vector_db_name = vector_search_request.vector_db_name if vector_search_request.vector_db_name else "default"
        vectordb = next((db for db in self.vector_dbs if db.name == vector_db_name), None)
        if not vectordb:
            raise ValueError(f"VectorDBItemBase with name {vector_db_name} not found")

        langchain_vector_db = self.get_vector_db(vector_db_name)

        # デバッグ出力
        logger.info('ベクトルDBの設定')
        logger.info(f'''
            name:{vectordb.name} vector_db_description:{vectordb.description} 
            VectorDBTypeString:{vectordb.get_vector_db_type_string()} VectorDBURL:{vectordb.vector_db_url} 
            CollectionName:{vectordb.collection_name}
            ChunkSize:{vectordb.chunk_size} 
            ''')


        logger.info(f'query: {vector_search_request.query}')
        search_kwargs = {}
        search_kwargs["k"] = vector_search_request.k
        if vector_search_request.filter:
            search_kwargs["filter"] = vector_search_request.filter

        logger.info(f'SearchKwargs:{search_kwargs}')
        documents =  await langchain_vector_db.vector_search(vector_search_request.query, search_kwargs)
        result_documents.extend(documents)

        return result_documents
    
