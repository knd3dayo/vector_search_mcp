import os, sys

from typing import Tuple, List, Any
import chromadb.config
from langchain_chroma.vectorstores import Chroma # type: ignore
import chromadb
from langchain_core.vectorstores import VectorStore # type: ignore

from vector_search_mcp.langchain.langchain_vector_db import LangChainVectorDB

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LangChainVectorDBChroma(LangChainVectorDB):
    
    def model_post_init(self, __context: Any) -> None:
        self.db = self._load()

    def _load(self) -> VectorStore:

        # ベクトルDB用のディレクトリが存在しない場合
        if not os.path.exists(self.vector_db_url):
            # ディレクトリを作成
            os.makedirs(self.vector_db_url)
            # ディレクトリが作成されたことをログに出力
            logger.info(f"create directory:{self.vector_db_url}")
        # params
        settings = chromadb.config.Settings(anonymized_telemetry=False)

        params: dict[str, Any]= {}
        params["client"] = chromadb.PersistentClient(path=self.vector_db_url, settings=settings)
        params["embedding_function"] = self.langchain_openai_client.get_embedding_client()
        params["collection_metadata"] = {
            "hnsw:space":"cosine", 
            "hnsw:construction_ef": 400, 
            "hnsw:search_ef": 200,
            "hnsw:M": 24,
        }
        # collectionが指定されている場合
        logger.info(f"collection_name:{self.collection_name}")
        if self.collection_name:
            params["collection_name"] = self.collection_name
                    
        db: VectorStore = Chroma(
            **params
            )
        return db

    def _get_document_ids_by_tag(self, name:str="", value:str="") -> Tuple[List, List]:
        ids=[]
        metadata_list = []
        doc_dict = self.db.get(where={name: value}) # type: ignore

        # デバッグ用
        logger.debug("_get_document_ids_by_tag doc_dict:", doc_dict)

        # vector idを取得してidsに追加
        ids.extend(doc_dict.get("ids", []))
        metadata_list.extend(doc_dict.get("metadata", []))

        return ids, metadata_list
