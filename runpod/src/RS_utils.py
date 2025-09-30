# rs_utils.py
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import snowflake.connector
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import holidays
import json
import time

logger = logging.getLogger(__name__)

class CSACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dim=256):
        super(CSACActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim * 2)
        )
        self.max_action = max_action
        self.action_dim = action_dim
        self.log_std_min = -10
        self.log_std_max = 2
        self.sparsity_weight = 0.01

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        mu_logstd = self.network(state)
        mu, log_std = mu_logstd[:, :self.action_dim], mu_logstd[:, self.action_dim:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action_raw = dist.rsample()
        action_squashed = torch.tanh(action_raw)
        action_normalized = torch.softmax(action_squashed * 2.0, dim=-1) * self.max_action
        return action_normalized

# def get_snowflake_connection(max_retries=3, retry_delay=30):
#     for attempt in range(1, max_retries + 1):
#         try:
#             with open("rsa_key_coach_snow.p8", "rb") as key_file:
#                 private_key = serialization.load_pem_private_key(
#                     key_file.read(),
#                     password=os.getenv("SNOWFLAKE_SSH_PASS").encode(),
#                     backend=default_backend()
#                 )
#             private_key_bytes = private_key.private_bytes(
#                 encoding=serialization.Encoding.DER,
#                 format=serialization.PrivateFormat.PKCS8,
#                 encryption_algorithm=serialization.NoEncryption()
#             )
#             logger.info(f"Attempt {attempt} to connect to Snowflake...")
#             conn = snowflake.connector.connect(
#                 user=os.getenv("SNOWFLAKE_USER"),
#                 private_key=private_key_bytes,
#                 account=os.getenv("SNOWFLAKE_ACCOUNT"),
#                 warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
#                 database=os.getenv("SNOWFLAKE_DATABASE"),
#                 schema=os.getenv("SNOWFLAKE_SCHEMA"),
#             )
#             logger.info("Snowflake connection established successfully.")
#             return conn
#         except snowflake.connector.errors.DatabaseError as e:
#             err_msg = str(e)
#             logger.warning(f"❌ Snowflake connection failed (attempt {attempt}): {err_msg}")
#             if "JWT token is invalid" in err_msg or "Failed to authenticate" in err_msg:
#                 if attempt < max_retries:
#                     logger.info(f"🔁 Waiting {retry_delay} seconds before retrying...")
#                     time.sleep(retry_delay)
#                 else:
#                     logger.error("❌ Max retries exceeded. Could not connect to Snowflake.")
#                     raise
#             else:
#                 raise

# conn = None

def get_snowflake_connection(max_retries=3, retry_delay=30):
    for attempt in range(1, max_retries + 1):
        try:
            # Load RSA private key directly from environment variable
            private_key_str = os.getenv("rsa_key_coach_snow")
            if not private_key_str:
                raise ValueError("❌ rsa_key_coach_snow not found in environment variables.")

            # Convert string with "\n" into proper PEM format
            private_key_bytes = private_key_str.encode().replace(b"\\n", b"\n")

            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=os.getenv("SNOWFLAKE_SSH_PASS").encode(),
                backend=default_backend()
            )

            private_key_der = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            logger.info(f"Attempt {attempt} to connect to Snowflake...")
            conn = snowflake.connector.connect(
                user=os.getenv("SNOWFLAKE_USER"),
                private_key=private_key_der,
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA"),
            )
            logger.info("✅ Snowflake connection established successfully.")
            return conn

        except snowflake.connector.errors.DatabaseError as e:
            err_msg = str(e)
            logger.warning(f"❌ Snowflake connection failed (attempt {attempt}): {err_msg}")
            if "JWT token is invalid" in err_msg or "Failed to authenticate" in err_msg:
                if attempt < max_retries:
                    logger.info(f"🔁 Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error("❌ Max retries exceeded. Could not connect to Snowflake.")
                    raise
            else:
                raise

conn = None


def get_or_refresh_connection():
    global conn
    if conn is None:
        conn = get_snowflake_connection()
    else:
        try:
            conn.cursor().execute("SELECT 1")
        except:
            logger.info("Connection lost, refreshing...")
            conn = get_snowflake_connection()
    return conn

def compute_dynamic_features(timestamp):
    us_holidays = holidays.US()
    time_encoder = OneHotEncoder(sparse_output=False)
    day_encoder = OneHotEncoder(sparse_output=False)
    
    time_categories = np.array([['Morning'], ['Afternoon'], ['Evening']])
    day_categories = np.array([['Monday'], ['Tuesday'], ['Wednesday'], ['Thursday'], ['Friday'], ['Saturday'], ['Sunday']])
    time_encoder.fit(time_categories)
    day_encoder.fit(day_categories)

    hour = timestamp.hour
    time_of_day = 'Morning' if 8 <= hour < 12 else 'Afternoon' if 12 <= hour < 16 else 'Evening'
    day_of_week = timestamp.strftime('%A')
    is_holiday = 1.0 if timestamp.date() in us_holidays else 0.0

    time_encoded = time_encoder.transform([[time_of_day]])[0]
    day_encoded = day_encoder.transform([[day_of_week]])[0]
    dynamic_features = np.concatenate([[is_holiday], time_encoded, day_encoded])
    return dynamic_features.astype(np.float32)

def encode_state_features(email, response_id):
    try:
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        query = """
            SELECT p.*
            FROM WORKSPACE.PUBLIC.COACH_CUSTOMER_PROFILE p
            LEFT JOIN WORKSPACE.PUBLIC.COACH_CONVERSIONS c
            ON p.RESPONSEID = c.RESPONSE_ID
            WHERE c.EMAIL = %s AND p.RESPONSEID = %s
        """
        cursor.execute(query, (email, response_id))
        profile_df = cursor.fetch_pandas_all()
        conversion_query = """
            SELECT * FROM WORKSPACE.PUBLIC.COACH_CONVERSIONS
            WHERE EMAIL = %s
        """
        cursor.execute(conversion_query, (email,))
        conversions_df = cursor.fetch_pandas_all()
        if profile_df.empty:
            logger.error(f"No profile found for email: {email}, response_id: {response_id}")
            return None
        expected_numerical_cols = ['WEIGHT', 'HEIGHT', 'AGE']
        numerical_cols = [col for col in expected_numerical_cols if col in profile_df.columns]
        if not numerical_cols:
            numerical_encoded = np.array([])
        else:
            numerical_data = profile_df[numerical_cols].copy()
            def convert_height(height):
                if pd.isna(height) or height == 'Unknown':
                    return np.nan
                if height == 'under 5 feet':
                    return 59
                try:
                    height = str(height).replace('"', '')
                    if "'" in height:
                        feet, inches = height.split("'")
                        return int(feet) * 12 + int(inches)
                    else:
                        return float(height)
                except:
                    logger.warning(f"Invalid height format: {height}")
                    return np.nan
            if 'HEIGHT' in numerical_data.columns:
                numerical_data['HEIGHT'] = numerical_data['HEIGHT'].apply(convert_height)
            numerical_data.fillna(numerical_data.mean(), inplace=True)
            numerical_min = numerical_data.min()
            numerical_max = numerical_data.max()
            denominator = numerical_max - numerical_min + 1e-6
            numerical_encoded = ((numerical_data - numerical_min) / denominator).values[0]

        categorical_cols = [
            'GENDER', 'BODYTYPE', 'BODYFAT', 'DIET', 'DIETRESTRICTIONS',
            'MACROS', 'PRIMARYGOAL', 'SECONDARYGOAL', 'ACTIVITYLEVEL',
            'ACTIVITY', 'INTENDEDUSE', 'FREQUENCY', 'MIXTIMING', 'HEALTHCONCERNS'
        ]
        categorical_dfs = []
        available_categorical_cols = [col for col in categorical_cols if col in profile_df.columns]
        
        for col in available_categorical_cols:
            features = set()
            for values in profile_df[col].dropna():
                for feature in [v.strip() for v in str(values).split(',') if v.strip()]:
                    features.add(feature)
            if features:
                features = sorted(features)
                dct = {f"{col}_{feature}": [1 if feature in str(values).split(',') else 0 for values in profile_df[col]] for feature in features}
                categorical_dfs.append(pd.DataFrame(dct, index=profile_df.index))
        categorical_combined = pd.concat(categorical_dfs, axis=1) if categorical_dfs else pd.DataFrame(index=profile_df.index)
        
        history = []
        if not conversions_df.empty:
            history.append(len(conversions_df))
            price_cols = [col for col in conversions_df.columns if 'PRICE' in col.upper()]
            history.append(conversions_df[price_cols[0]].sum() if price_cols else 0)
            date_cols = [col for col in conversions_df.columns if any(date_word in col.upper() for date_word in ['CREATE', 'DATE', 'TIME'])]
            if date_cols:
                try:
                    max_date = pd.to_datetime(conversions_df[date_cols[0]]).max()
                    days_since = (datetime.utcnow() - max_date).days
                    history.append(days_since)
                except:
                    history.append(0)
            else:
                history.append(0)
        else:
            history = [0, 0, 0]

        createdon_col = 'CREATEDON'
        if createdon_col in profile_df.columns and not pd.isna(profile_df[createdon_col].iloc[0]):
            createdon = pd.to_datetime(profile_df[createdon_col].iloc[0])
        else:
            createdon = datetime.now()
            logger.warning(f"No valid CREATEDON found; using current time: {createdon}")
        dynamic_part = compute_dynamic_features(createdon)

        numerical_part = numerical_encoded if len(numerical_encoded) > 0 else np.array([])
        categorical_part = categorical_combined.values[0] if len(categorical_combined) > 0 else np.array([])
        history_part = np.array(history, dtype=np.float32)

        state_parts = [part for part in [numerical_part, categorical_part, history_part, dynamic_part] if len(part) > 0]
        if state_parts:
            state = np.concatenate(state_parts).astype(np.float32)
        else:
            logger.error("No valid state features could be extracted")
            return None

        expected_dim = 120
        if len(state) != expected_dim:
            logger.warning(f"State dimension is {len(state)}, expected {expected_dim}. Padding/truncating...")
            if len(state) < expected_dim:
                state = np.pad(state, (0, expected_dim - len(state)), mode='constant')
            else:
                state = state[:expected_dim]
        return state
    except Exception as e:
        logger.error(f"State encoding error: {str(e)}")
        return None


def format_recommendation(action, variant_ids, categories):
    protein_mix = []
    total_percent = sum(action)
    if total_percent > 0:
        action_normalized = [float(a) / total_percent for a in action]
    else:
        action_normalized = [float(a) for a in action]
    for i, (variant_id, category) in enumerate(zip(variant_ids, categories)):
        item_name = get_item_name(variant_id)
        if action_normalized[i] > 0.05:
            protein_mix.append({
                "variant_id": str(variant_id),
                "percent": float(action_normalized[i] * 100),
                "category": str(category),
                "item_name": str(item_name)
            })
    return {"protein_mix": protein_mix}

def get_item_name(variant_id):
    try:
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT PRODUCTNAME FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_CLASSIFICATION WHERE VARIANTID = %s",
            (variant_id,)
        )
        result = cursor.fetchone()
        item_name = result[0] if result else "unknown"
        return str(item_name)
    except Exception as e:
        logger.error(f"Error fetching item name for variant_id {variant_id}: {str(e)}")
        return "unknown"

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)