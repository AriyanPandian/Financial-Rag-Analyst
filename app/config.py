from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    data_dir: Path
    max_upload_bytes: int
    api_access_token: str

    @classmethod
    def from_env(cls) -> "Settings":
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "50"))
        if max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be a positive integer.")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            data_dir=Path(os.getenv("DATA_DIR", "./data")).expanduser(),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            api_access_token=os.getenv("API_ACCESS_TOKEN", "").strip(),
        )
