import os

from mlflow.store.db import utils as mlflow_db_utils


def main() -> None:
    database_uri = os.environ["MLFLOW_BACKEND_STORE_URI"]
    engine = mlflow_db_utils.create_sqlalchemy_engine_with_retry(database_uri)

    if mlflow_db_utils._all_tables_exist(engine):
        print("Upgrading the existing MLflow database schema.")
        mlflow_db_utils._upgrade_db(engine)
    else:
        print("Initializing a fresh MLflow database schema.")
        mlflow_db_utils._initialize_tables(engine)


if __name__ == "__main__":
    main()
