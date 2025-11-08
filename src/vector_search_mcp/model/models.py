from pydantic import BaseModel, Field, field_validator
from typing import Optional, ClassVar, Any
import os


import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class EmbeddingData(BaseModel):
    model: str
    source_id: str
    folder_path: str 
    description: str = ""
    content: str
    source_path: str = ""
    image_url: str = ""
    vector_db_name: str = ""  # 追加: ベクトルDB名



class VectorSearchRequest(BaseModel):
    vector_db_name: str = Field(
        default="default",
        description="Name of the vector search request. This is used to identify the request in the system."
    )
    query: Optional[str] = Field(
        default="",
        description="The query string to search for in the vector database. This is the main input for the vector search."
    )
    k: int = Field(
        default=5,
        description="The number of results to return from the vector search."
    )
    filter : dict[str, Any] = Field(
        default_factory=dict,
        description="Filter criteria to apply to the vector search."
    )

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

