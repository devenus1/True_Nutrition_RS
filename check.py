import snowflake.connector
import pyotp

# Replace these with your actual Snowflake credentials and configuration
conn_params = {
    'user': 'HESUN',  # e.g., 'john.doe@company.com'
    'password': 'Intrinsic-tech@dev-venus@3964081',
    'account': 'ljrrcwh-pkb82607',  # e.g., 'xyz12345' or 'xyz12345.us-east-1.aws'
    'warehouse': 'RECOMMENDER',  # e.g., 'COMPUTE_WH'
    'database': 'INGEST_DATA',  # e.g., 'INGEST_DATA'
    'schema': 'TRUENUTRITION_BUILD',  # e.g., 'TRUENUTRITION_BUILD'
    'passcode': pyotp.TOTP('2XCKNL6UVO2GPZ5CACFGXCMDB5Z2USD3').now()
}

try:
    conn = snowflake.connector.connect(**conn_params)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")


