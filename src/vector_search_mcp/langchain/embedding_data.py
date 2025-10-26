from pydantic import BaseModel

import vector_search_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class EmbeddingData(BaseModel):
    name: str
    model: str
    source_id: str
    folder_path: str 
    description: str = ""
    content: str
    source_path: str = ""
    image_url: str = ""
