
import uuid
from typing import Tuple, List, Any, Union, Optional
from collections import defaultdict
import asyncio
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import RateLimitError

from vector_search_mcp.langchain.langchain_client import LangChainOpenAIClient
from vector_search_mcp.langchain.langchain_doc_store import SQLDocStore
from vector_search_mcp.langchain.embedding_data import EmbeddingData

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class LangChainVectorDB(BaseModel):
    """
    LangChainのベクトルDBを利用するための基底クラス。
    """
    langchain_openai_client: LangChainOpenAIClient = Field(..., description="LangChain OpenAI Client")
    vector_db_url: str = Field(..., description="Vector DBのURL")
    collection_name: str = Field(default="", description="コレクション名")
    doc_store_url: str = Field(default="", description="MultiVectorRetrieverを利用する場合のDocStoreのURL")
    chunk_size: int = Field(default=1000, description="テキストを分割するチャンクサイズ")
    use_multi_vector_retriever: bool = Field(default=False, description="MultiVectorRetrieverを利用するかどうか")
    parent_chunk_size: int = Field(default=4000, description="親データのチャンクサイズ(MultiVectorRetrieverを利用する場合)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db : Union[VectorStore, None] = Field(default=None, description="VectorStoreのインスタンス")
    doc_store: Union[SQLDocStore, None] = Field(default=None, description="SQLDocStoreのインスタンス")


    # document_idのリストとmetadataのリストを返す
    def _get_document_ids_by_tag(self, name: str = "", value: str = "") -> Tuple[List[str], List[dict[str, Any]]]:
        # 未実装例外をスロー
        raise NotImplementedError("Not implemented")

    async def _delete(self, doc_ids:list=[]):
        if len(doc_ids) == 0:
            return
        if self.db is None:
            raise ValueError("db is None")

        await self.db.adelete(ids=doc_ids)

        return len(doc_ids)    

    def _delete_collection(self):
        # self.dbがdelete_collectionメソッドを持っている場合はそれを呼び出す
        if hasattr(self.db, "delete_collection"):
            self.db.delete_collection() # type: ignore


    async def add_document(self, data: EmbeddingData):

        if self.db is None:
            raise ValueError("db is None")
       # テキストをサニタイズ
        page_content = self._sanitize_text(data.content)
 
        doc_id_text_list: list[tuple[str, str]] = []
        # doc_store_urlが指定されている場合は、page_contentをparent_chunk_sizeで分割, doc_idとtextのタプルを作成
        doc_store: Optional[SQLDocStore] = None
        if self.doc_store_url:
            doc_store = SQLDocStore(self.doc_store_url)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size)
            text_list = text_splitter.split_text(page_content)
            for text in text_list:
                doc_id = str(uuid.uuid4())
                doc_id_text_list.append((doc_id, text))
        else:
            # doc_store_urlが指定されていない場合は、page_contentとdoc_idのタプルを作成
            doc_id = str(uuid.uuid4())
            doc_id_text_list.append((doc_id, page_content))

        # doc_id_text_listの要素をループして、Documentを作成
        for doc_id, text in doc_id_text_list:
            # metadataをコピーしてdoc_idを設定
            metadata_copy = await LangChainVectorDB.create_metadata(data)
            metadata_copy["doc_id"] = doc_id

            # Documentを作成
            document = Document(
                page_content=text,
                metadata=metadata_copy
            )
            await self.add_doucment_with_retry(self.db, [document])

            if doc_store is not None:
                # doc_store_urlが指定されている場合は、doc_storeに保存
                param = []
                param.append((doc_id, document))
                await doc_store.amset(param)

    # テキストをサニタイズする
    def _sanitize_text(self, text: str) -> str:
        # textが空の場合は空の文字列を返す
        if not text or len(text) == 0:
            return ""
        import re
        # 1. 複数の改行を1つの改行に変換
        text = re.sub(r'\n+', '\n', text)
        # 2. 複数のスペースを1つのスペースに変換
        text = re.sub(r' +', ' ', text)

        return text

    ########################################
    # パブリック
    ########################################

    def delete_collection(self):
        # ベクトルDB固有の削除メソッドを呼び出してコレクションを削除
        self._delete_collection()

    async def delete_folder(self, folder_path: str):
        # ベクトルDB固有のvector id取得メソッドを呼び出し。
        vector_ids, _ = self._get_document_ids_by_tag("folder_path", folder_path)

        # vector_idsが空の場合は何もしない
        if len(vector_ids) == 0:
            return 0

        # DocStoreから削除
        if self.doc_store_url and self.doc_store is not None:
            await self.doc_store.amdelete(vector_ids)

        # ベクトルDB固有の削除メソッドを呼び出し
        await self._delete(vector_ids)

    async def delete_document(self, source_id: str):
        # ベクトルDB固有のvector id取得メソッドを呼び出し。
        doc_ids, _ = self._get_document_ids_by_tag("source_id", source_id)

        # vector_idsが空の場合は何もしない
        if len(doc_ids) == 0:
            return 0

        # DocStoreから削除
        if self.doc_store_url and self.doc_store is not None:
            await self.doc_store.amdelete(doc_ids)

        # ベクトルDB固有の削除メソッドを呼び出し
        await self._delete(doc_ids)

    async def update_embeddings(self, params: EmbeddingData):
        
        # 既に存在するドキュメントを削除
        await self.delete_document(params.source_id)
        # ドキュメントを格納する。
        await self.add_document(params)

    # RateLimitErrorが発生した場合は、指数バックオフを行う
    async def add_doucment_with_retry(self, vector_db: VectorStore, documents: list[Document], max_retries: int = 5, delay: float = 1.0):
        for attempt in range(max_retries):
            try:
                await vector_db.aadd_documents(documents=documents)
                return
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"RateLimitError: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Max retries reached. Failed to add documents: {e}")
                    break
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                break

    async def vector_search(self, query: str, search_kwargs: dict, return_parent: bool = True) -> List[Document]:
        """
        ベクトルDBからドキュメントを検索する。
        :param query: 検索クエリ
        :param search_kwargs: 検索キーワード
        :param return_parent: Trueの場合で、MultiVectorRetrieverを利用している場合は、親ドキュメントも返す
        """
        if self.db is None:
            raise ValueError("db is None")

        docs_and_scores = self.db.similarity_search_with_relevance_scores(query, **search_kwargs)
        # documentのmetadataにscoreを追加
        doc_ids: set[str] = set()
        documents: List[Document] = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            documents.append(doc)
            doc_id = doc.metadata.get("doc_id", None)
            if doc_id is not None:
                doc_ids.add(doc_id)

        if self.doc_store_url and return_parent:
            doc_store = SQLDocStore(self.doc_store_url)
            # doc_store_urlが指定されている場合は、doc_storeからドキュメントを取得
            parent_docs = await doc_store.amget(list(doc_ids))
                                
            result_docs: List[Document] = []
            for parent_doc in parent_docs:
                if isinstance(parent_doc, Document):
                    # parent_docのmetadataにscoreを追加
                    parent_doc.metadata["score"] = 0
                    # documentsの中から同じdoc_idのドキュメントを探してsub_docsに追加
                    sub_docs = [doc.model_dump() for doc in documents if doc.metadata.get("doc_id") == parent_doc.metadata.get("doc_id")]
                    parent_doc.metadata["sub_docs"] = sub_docs
                    # parent_docをresult_docsに追加
                    result_docs.append(parent_doc)

            # result_docsを返す
            return result_docs
        else:
            return documents

    @classmethod
    async def create_metadata(cls, embedding_data: EmbeddingData) -> dict[str, Any]:
        logger.info(f"folder_path:{embedding_data.folder_path}")

        metadata = {
            "folder_path": embedding_data.folder_path,
            "source_path": embedding_data.source_path,
            "description": embedding_data.description,
            "source_id": embedding_data.source_id,
            "source_type": 0,
            "score": 0
        }
        return metadata

    
