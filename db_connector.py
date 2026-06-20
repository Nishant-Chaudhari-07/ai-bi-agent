import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
load_dotenv()

DB_PATH = "data/demo.db"
CSV_PATH = "data/insurance_claims.csv"


def get_engine():
    """
    Auto-detects which database to use.
    If Snowflake credentials + private key exist → Snowflake (key-pair auth, no MFA).
    Otherwise → SQLite (demo mode).
    """
    snowflake_user = os.getenv("SNOWFLAKE_USER")
    snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")

    if snowflake_user and snowflake_account and private_key_path:
        print("MODE: Snowflake (live - key pair auth)")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend

        with open(private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        from snowflake.sqlalchemy import URL
        engine = create_engine(
            URL(
                account=snowflake_account,
                user=snowflake_user,
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            ),
            connect_args={"private_key": private_key_bytes}
        )
        return engine, "snowflake"
    else:
        print("MODE: SQLite (demo)")
        engine = create_engine(f"sqlite:///{DB_PATH}")
        return engine, "sqlite"


def load_csv_to_sqlite():
    """Loads the insurance claims CSV into SQLite demo database."""
    print("Loading CSV into SQLite...")
    df = pd.read_csv(CSV_PATH)

    # Clean up columns
    df.columns = [c.replace("-", "_").replace(" ", "_").lower() for c in df.columns]
    df = df.drop(columns=["_c39"], errors="ignore")  # drop empty trailing column

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("insurance_claims", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into SQLite table 'insurance_claims'")


def load_csv_to_snowflake():
    """Loads the insurance claims CSV into Snowflake."""
    print("Loading CSV into Snowflake...")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.replace("-", "_").replace(" ", "_").lower() for c in df.columns]
    df = df.drop(columns=["_c39"], errors="ignore")

    engine, _ = get_engine()
    df.to_sql("insurance_claims", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into Snowflake table 'insurance_claims'")


if __name__ == "__main__":
    # Load into SQLite always (for demo mode)
    load_csv_to_sqlite()

    # Load into Snowflake if credentials exist
    snowflake_user = os.getenv("SNOWFLAKE_USER")
    if snowflake_user:
        try:
            load_csv_to_snowflake()
        except Exception as e:
            print(f"Snowflake load skipped: {e}")
    else:
        print("No Snowflake credentials found - skipping Snowflake load.")

    print("\nDatabase setup complete.")