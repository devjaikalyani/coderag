from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama3-70b-8192", env="GROQ_MODEL")

    # Embedding / Reranker
    embedding_model: str = Field("microsoft/codebert-base", env="EMBEDDING_MODEL")
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")

    # FAISS
    faiss_index_path: Path = Field("data/processed/faiss_index", env="FAISS_INDEX_PATH")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    top_k_retrieve: int = Field(20, env="TOP_K_RETRIEVE")
    top_k_rerank: int = Field(5, env="TOP_K_RERANK")

    # MLflow
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field("coderag", env="MLFLOW_EXPERIMENT")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # Faithfulness
    faithfulness_threshold: float = Field(0.5, env="FAITHFULNESS_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
