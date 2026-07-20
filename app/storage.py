import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, MetaData, String, Table, Text, create_engine, insert
from sqlalchemy.engine import Engine

from app.config import Settings

metadata = MetaData()

analysis_logs = Table(
    "analysis_logs",
    metadata,
    # sqlite autoincrement also works with this Integer PK.
    # The public analysis_id is stored separately.
    # This keeps the schema simple across SQLite and PostgreSQL.
    # SQLAlchemy will emit SERIAL/IDENTITY where supported.
    __import__("sqlalchemy").Column("id", Integer, primary_key=True),
    __import__("sqlalchemy").Column("analysis_id", String(64), nullable=False),
    __import__("sqlalchemy").Column("consultation_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("branch_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("client_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("payload_json", Text, nullable=False),
    __import__("sqlalchemy").Column("created_at", DateTime(timezone=True), nullable=False),
)

recommendation_logs = Table(
    "recommendation_logs",
    metadata,
    __import__("sqlalchemy").Column("id", Integer, primary_key=True),
    __import__("sqlalchemy").Column("branch_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("client_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("score", Integer, nullable=False),
    __import__("sqlalchemy").Column("payload_json", Text, nullable=False),
    __import__("sqlalchemy").Column("created_at", DateTime(timezone=True), nullable=False),
)

outcome_logs = Table(
    "outcome_logs",
    metadata,
    __import__("sqlalchemy").Column("id", Integer, primary_key=True),
    __import__("sqlalchemy").Column("object_type", String(32), nullable=False),
    __import__("sqlalchemy").Column("object_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("branch_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("staff_action", String(32), nullable=False),
    __import__("sqlalchemy").Column("final_outcome", String(64), nullable=True),
    __import__("sqlalchemy").Column("notes", Text, nullable=True),
    __import__("sqlalchemy").Column("created_at", DateTime(timezone=True), nullable=False),
)


def create_storage_engine(settings: Settings) -> Engine:
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, future=True, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def save_analysis(engine: Engine, request: Any, response: Any) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(analysis_logs).values(
                analysis_id=response.analysis_id,
                consultation_id=request.consultation_id,
                branch_id=request.branch_id,
                client_id=request.client_id,
                payload_json=response.model_dump_json(),
                created_at=_now(),
            )
        )


def save_recommendations(engine: Engine, branch_id: str, recommendations: list[Any]) -> None:
    if not recommendations:
        return
    with engine.begin() as connection:
        connection.execute(
            insert(recommendation_logs),
            [
                {
                    "branch_id": branch_id,
                    "client_id": recommendation.client_id,
                    "score": recommendation.score,
                    "payload_json": recommendation.model_dump_json(),
                    "created_at": _now(),
                }
                for recommendation in recommendations
            ],
        )


def save_outcome(engine: Engine, outcome: Any) -> None:
    payload = outcome.model_dump()
    with engine.begin() as connection:
        connection.execute(
            insert(outcome_logs).values(
                object_type=payload["object_type"],
                object_id=payload["object_id"],
                branch_id=payload["branch_id"],
                staff_action=payload["staff_action"],
                final_outcome=payload["final_outcome"],
                notes=payload["notes"],
                created_at=_now(),
            )
        )


def serialize_for_crm(model: Any) -> dict[str, Any]:
    return json.loads(model.model_dump_json())


def _now() -> datetime:
    return datetime.now(timezone.utc)

