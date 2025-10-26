from typing import List, Union, Optional, ClassVar
from pydantic import BaseModel, Field
from typing import Optional, List
from typing import Optional
from typing import Optional, Union, List
from typing import Optional, List, Dict, Any, Union
from vector_search_mcp.langchain.langchain_util import VectorDBItemBase
import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class VectorSearchRequest(BaseModel):
    name: str = Field(
        default="default",
        description="Name of the vector search request. This is used to identify the request in the system."
    )
    query: Optional[str] = Field(
        default="",
        description="The query string to search for in the vector database. This is the main input for the vector search."
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="The model to use for vector embedding. Default is 'text-embedding-ada-002'."
    )
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    vector_db_item: Optional["VectorDBItemBase"] = None

