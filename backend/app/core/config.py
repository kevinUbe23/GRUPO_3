from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COBRANZAS_", env_file=".env")

    app_name: str = "Sistema de priorizacion de cobranzas"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_origin_regex: str | None = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    project_root: Path = Path(__file__).resolve().parents[3]
    database_url: str | None = None

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = self.project_root / "backend" / "data" / "cobranzas.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def generated_data_dir(self) -> Path:
        return self.project_root / "01_generacion" / "data"

    @property
    def prep_outputs_dir(self) -> Path:
        return self.project_root / "03_preparacion" / "outputs"

    @property
    def model_outputs_dir(self) -> Path:
        return self.project_root / "04_evaluacion_modelos_ia" / "outputs"

    @property
    def model_path(self) -> Path:
        return self.model_outputs_dir / "best_model_artifact.joblib"

    @property
    def feature_schema_path(self) -> Path:
        return self.model_outputs_dir / "model_feature_schema.csv"

    @property
    def prepared_features_path(self) -> Path:
        return self.prep_outputs_dir / "features_ml_prepared.csv"

    @property
    def frontend_segments_path(self) -> Path:
        return self.model_outputs_dir / "frontend_customer_segments.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
