# src/mcp_agent/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class AgentConfig(BaseSettings):
    llm_endpoint_name: str = Field(default="databricks-claude-sonnet-4-5", alias="LLM_ENDPOINT_NAME")
    databricks_host: str = Field(..., env="DATABRICKS_HOST")
    mcp_system_ai_url: str = Field(..., env="MCP_SYSTEM_AI_URL")
    environment: str = Field(default="development")

    class Config:
        env_file = ".env"
        populate_by_name = True