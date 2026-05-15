from __future__ import annotations

from app.db.database import Base, SessionLocal, engine
from app.services.import_service import reset_and_import_seed_data


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        result = reset_and_import_seed_data(db)
    print(result)


if __name__ == "__main__":
    main()
