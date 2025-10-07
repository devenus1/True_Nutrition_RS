import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
import requests
import json as json_module
from pytz import timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from threading import Lock

import re
import traceback
import cvxpy as cp
import logging
from logging.handlers import RotatingFileHandler
from fastapi import Request
from fastapi.responses import JSONResponse
import redis
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import no_grad
import pyotp
import collections
import json
from datetime import datetime
from collections import OrderedDict
from pydantic import BaseModel
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import holidays  # For US holidays
import os
from dotenv import load_dotenv
import redis
# from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("logs/application.log", maxBytes=5*1024*1024, backupCount=7),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

load_dotenv() 

# Initialize Redis client
redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

def get_from_cache(response_id):
    cached_data = redis_client.get(response_id)
    if cached_data:
        return json.loads(cached_data)  # Deserialize JSON to dictionary
    return None

def store_in_cache(response_id, data):  
    redis_client.set(response_id, json.dumps(data))

def get_snowflake_connection(max_retries=3, retry_delay=30):
    for attempt in range(1, max_retries + 1):
        try:
            # Load RSA private key directly from environment variable
            private_key_str = os.getenv("rsa_key_coach_snow")
            if not private_key_str:
                raise ValueError(" rsa_key_coach_snow not found in environment variables.")

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
            logger.info(" Snowflake connection established successfully.")
            return conn

        except snowflake.connector.errors.DatabaseError as e:
            err_msg = str(e)
            logger.warning(f"Snowflake connection failed (attempt {attempt}): {err_msg}")
            if "JWT token is invalid" in err_msg or "Failed to authenticate" in err_msg:
                if attempt < max_retries:
                    logger.info(f" Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries exceeded. Could not connect to Snowflake.")
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

class recommendationRequest(BaseModel):
    email: str
    response_id: str





app = FastAPI()

security = HTTPBearer()




actor = None
variant_ids = None
categories = None

@app.on_event("startup")
async def startup_event():
    global actor, variant_ids, categories
    try:
        # Load variant data from Snowflake
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        # cursor.execute("SELECT VARIANTID, CATEGORY FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_NUTRITION_DATA WHERE CATEGORY = 'Core protein'")
        # cursor.execute("SELECT VARIANTID, CATEGORY FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_NUTRITION_DATA ")
        cursor.execute("""
            SELECT DISTINCT 
                vn.VARIANTID, 
                vn.CATEGORY
            FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_NUTRITION_DATA vn
            INNER JOIN (
                SELECT DISTINCT VARIANTID
                FROM INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BASE_RESULT
                WHERE VARIANTID IS NOT NULL
            ) br ON CAST(vn.VARIANTID AS STRING) = CAST(br.VARIANTID AS STRING)
            ORDER BY vn.VARIANTID
            LIMIT 43 
        """)
        #match the output dimension of the machine learning model.
        variant_df = cursor.fetch_pandas_all()
        variant_ids = [str(v) for v in variant_df['VARIANTID'].tolist()]  # Convert to strings
        categories = [str(c) for c in variant_df['CATEGORY'].tolist()]    # Convert to strings
        logger.info("Variant data loaded successfully")
        
        # Initialize and load the actor model
        state_dim = 120 
        action_dim = len(variant_ids)  # Number of variants for action space
        actor = CSACActor(state_dim=state_dim, action_dim=action_dim)
        with open('improved_csac_actor.pth', 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
            actor.load_state_dict(state_dict)
        actor.eval()  # Set to evaluation mode
        logger.info(f"Actor model loaded successfully with state_dim={state_dim}, action_dim={action_dim}")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"Failed to initiate: {str(e)}")
    
def convert_height_to_inches(height_str):
    try:
        if height_str == "Under 5 feet":
            height_str = "4'11\""
        if height_str == "Over 7 feet":
            height_str = "7'1\""
        feet, inches = height_str.split("'")
        inches = inches.replace('"', '').strip()
        return int(feet) * 12 + int(inches)
    except (ValueError, AttributeError):
        return None
    except Exception as e:
        logger.error(f"Error converting height to inches: {e}")
        return None

    

def extract_responses(typeform_item):
    response_dict = {
            "EMAIL": None,
            "GENDER": None,
            "AGE": None,
            "HEIGHT": None,
            "WEIGHT": None,
            "BODYTYPE": None,
            "BODYFAT": None,
            "DIET": None,
            "DIETRESTRICTIONS": None,
            "MACROS": None,
            "PRIMARYGOAL": None,
            "SECONDARYGOAL": None,
            "ACTIVITY": None,
            "ACTIVITYLEVEL": None,
            "INTENDEDUSE": None,
            "FREQUENCY": None,
            "MIXTIMING": None,
            "HEALTHCONCERNS": None,
            "CREATEDON": pd.to_datetime(typeform_item['submitted_at'])
        }

    for answer in typeform_item['answers']:
            field_ref = answer['field']['ref']
            if field_ref == settings.field_ref_email:
                response_dict['EMAIL'] = answer.get('email', None)
            elif field_ref == settings.field_ref_gender:
                response_dict['GENDER'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_age:
                response_dict['AGE'] = answer.get('number', None)
            elif field_ref == settings.field_ref_height_inches:
                height_str = answer.get('text', None)
                response_dict['HEIGHT'] = convert_height_to_inches(height_str)
            elif field_ref == settings.field_ref_weight:
                response_dict['WEIGHT'] = answer.get('number', None)
            elif field_ref == settings.field_ref_body_type:
                response_dict['BODYTYPE'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_body_fat:
                response_dict['BODYFAT'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_diet:
                response_dict['DIET'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_diet_restrictions:
                response_dict['DIETRESTRICTIONS'] = ', '.join(answer.get('choices', {}).get('labels', []))
            elif field_ref == settings.field_ref_macros:
                response_dict['MACROS'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_goal_primary:
                response_dict['PRIMARYGOAL'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_goal_secondary:
                response_dict['SECONDARYGOAL'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_activity:
                response_dict['ACTIVITY'] = ', '.join(answer.get('choices', {}).get('labels', []))
            elif field_ref == settings.field_ref_activity_level:
                response_dict['ACTIVITYLEVEL'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_mix_intended_use:
                response_dict['INTENDEDUSE'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_mix_frequency:
                response_dict['FREQUENCY'] = answer.get('choice', {}).get('label', None)
            elif field_ref == settings.field_ref_mix_timing:
                response_dict['MIXTIMING'] = ', '.join(answer.get('choices', {}).get('labels', []))
            elif field_ref == settings.field_ref_health_issues:
                response_dict['HEALTHCONCERNS'] = ', '.join(answer.get('choices', {}).get('labels', []))

    
    current_time = datetime.now()
  

    # Add timestamp and day of week to JSON
    response_dict['TNLINEITEMID'] = 0.0
    response_dict['MIN PURCHASE TIMESTAMP'] = current_time
   

    user_response_df = pd.DataFrame([response_dict])
    

    return user_response_df

def compute_dynamic_features(timestamp):
    """
    Compute dynamic features based on timestamp: is_holiday (1), time_of_day (3), day_of_week (7) = 11 features.
    """
    us_holidays = holidays.US()
    time_encoder = OneHotEncoder(sparse_output=False)
    day_encoder = OneHotEncoder(sparse_output=False)
    
    # Predefined categories
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

def encode_state_features(user_response_df):
    try:    
        expected_numerical_cols = ['WEIGHT', 'HEIGHT', 'AGE']
        numerical_cols = [col for col in expected_numerical_cols if col in user_response_df.columns]
        if not numerical_cols:
            logger.warning("No expected numerical columns found in profile data")
            numerical_encoded = np.array([])
        else:
            numerical_data = user_response_df[numerical_cols].copy()

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
            numerical_encoded = (numerical_data - numerical_min) / denominator
        categorical_cols = [
            'GENDER', 'BODYTYPE', 'BODYFAT', 'DIET', 'DIETRESTRICTIONS',
            'MACROS', 'PRIMARYGOAL', 'SECONDARYGOAL', 'ACTIVITYLEVEL',
            'ACTIVITY', 'INTENDEDUSE', 'FREQUENCY', 'MIXTIMING', 'HEALTHCONCERNS'
        ]
        categorical_dfs = []
        available_categorical_cols = [col for col in categorical_cols if col in user_response_df.columns]
        
        for col in available_categorical_cols:
            features = set()
            for values in user_response_df[col].dropna():
                for feature in [v.strip() for v in str(values).split(',') if v.strip()]:
                    features.add(feature)
            if features:
                features = sorted(features)
                logger.info(f"Column {col} has features: {features}")
                dct = {f"{col}_{feature}": [1 if feature in str(values).split(',') else 0 for values in user_response_df[col]] for feature in features}
                categorical_dfs.append(pd.DataFrame(dct, index=user_response_df.index))
        categorical_combined = pd.concat(categorical_dfs, axis=1) if categorical_dfs else pd.DataFrame(index=user_response_df.index)
        

        # Compute dynamic features using CREATEDON timestamp
        createdon_col = 'CREATEDON'
        if createdon_col in user_response_df.columns and not pd.isna(user_response_df[createdon_col].iloc[0]):
            createdon = pd.to_datetime(user_response_df[createdon_col].iloc[0])
        else:
            createdon = datetime.now()
            logger.warning(f"No valid CREATEDON found; using current time: {createdon}")
        dynamic_part = compute_dynamic_features(createdon)
        logger.info(f"Dynamic features length: {len(dynamic_part)}")
        numerical_part = numerical_encoded.values[0] if len(numerical_encoded) > 0 else np.array([])
        categorical_part = categorical_combined.values[0] if len(categorical_combined) > 0 else np.array([])
        
        logger.info(f"Numerical features: {len(numerical_part)}, Categorical features: {len(categorical_part)}, Dynamic features: {len(dynamic_part)}")
        state_parts = [part for part in [numerical_part, categorical_part, dynamic_part] if len(part) > 0]
        if state_parts:
            state = np.concatenate(state_parts, dtype=np.float32)
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
        logger.error(f"State encoding error: {str(e)}\n{traceback.format_exc()}")
        return None
    



def format_recommendation(action, variant_ids, categories):
    protein_mix = []
    total_percent = sum(action)
    if total_percent > 0:
        action_normalized = [float(a) / total_percent for a in action]
    else:
        action_normalized = [float(a) for a in action]
    for i, (variant_id, category) in enumerate(zip(variant_ids, categories)):
        logger.info(f"Variant ID type: {type(variant_id)}, Category type: {type(category)}")
        item_name = get_item_name(variant_id)
        logger.info(f"Item name type: {type(item_name)}")
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
        return str(item_name)  # Ensure string type
    except Exception as e:
        logger.error(f"Error fetching item name for variant_id {variant_id}: {str(e)}")
        return "unknown"

def custom_serializer(obj):
    if isinstance(obj, collections.OrderedDict):
        return dict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)

AUTHORIZATION_TOKEN = settings.authorization_token
FORM_ID = settings.form_id
TYPEFORM_API_TOKEN = settings.typeform_api_token

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTHORIZATION_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )
    



@app.get("/get-recommendations/{response_id}")
async def get_recommendations(response_id: str, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    user_email, user_name, survey_date = None, None, None

    # try:
        
    logger.info(f"Fetching recommendations for response_id: {response_id}")

    recommendations = get_from_cache(response_id)
    if recommendations:
        logger.info(f"Cache hit for response_id: {response_id}")
        return recommendations

    try:

        
        url = f"https://api.typeform.com/forms/{FORM_ID}/responses?included_response_ids={response_id}"
        headers = {
            "Authorization": f"Bearer {TYPEFORM_API_TOKEN}",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        typeform_data = response.json()

        if "items" not in typeform_data or not typeform_data["items"]:
            raise HTTPException(status_code=422, detail="No responses found in Typeform")
            
        typeform_item = typeform_data["items"][0]
        survey_date = typeform_item['submitted_at']
        for answer in typeform_item['answers']:
            field_ref = answer['field']['ref']
            if field_ref == settings.field_ref_firstname:
                user_name = answer.get('text', None)
            if field_ref == settings.field_ref_email:
                user_email = answer.get('email', None)

        user_df = extract_responses(typeform_item)
        state = encode_state_features(user_df)

        if state is None:
            raise HTTPException(status_code=400, detail="Failed to encode state features")
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action = [float(x) for x in action]
        recommendation = format_recommendation(action, variant_ids, categories)

        # # Extract profile from Typeform for response
        # profile_list = [
        #     {
        #         "gender": typeform_item.get("GENDER"),
        #         "age": typeform_item.get("AGE"),
        #         "height_inches": typeform_item.get("HEIGHT"),
        #         "weight": typeform_item.get("WEIGHT"),
        #         "body_type": typeform_item.get("BODYTYPE"),
        #         "body_fat": typeform_item.get("BODYFAT"),
        #         "diet": typeform_item.get("DIET"),
        #         "diet_restrictions": typeform_item.get("DIETRESTRICTIONS"),
        #         "macros": typeform_item.get("MACROS"),
        #         "goal_primary": typeform_item.get("PRIMARYGOAL"),
        #         "goal_secondary": typeform_item.get("SECONDARYGOAL"),
        #         "activity": typeform_item.get("ACTIVITY"),
        #         "activity_level": typeform_item.get("ACTIVITYLEVEL"),
        #         "mix_intended_use": typeform_item.get("INTENDEDUSE"),
        #         "mix_frequency": typeform_item.get("FREQUENCY"),
        #         "mix_timing": typeform_item.get("MIXTIMING"),
        #         "health_issues": typeform_item.get("HEALTHCONCERNS"),
        #     }
        # ]

        # Extract profile from the processed DataFrame
        profile_data = user_df.iloc[0]  # Get first row
        profile_list = [
            {
                "gender": profile_data.get("GENDER"),
                # "age": profile_data.get("AGE"),
                # "height_inches": profile_data.get("HEIGHT"),
                # "weight": profile_data.get("WEIGHT"),
                "age": int(profile_data.get("AGE")) if pd.notna(profile_data.get("AGE")) else None,
                "height_inches": int(profile_data.get("HEIGHT")) if pd.notna(profile_data.get("HEIGHT")) else None,
                "weight": float(profile_data.get("WEIGHT")) if pd.notna(profile_data.get("WEIGHT")) else None,
                "body_type": profile_data.get("BODYTYPE"),
                "body_fat": profile_data.get("BODYFAT"),
                "diet": profile_data.get("DIET"),
                "diet_restrictions": profile_data.get("DIETRESTRICTIONS"),
                "macros": profile_data.get("MACROS"),
                "goal_primary": profile_data.get("PRIMARYGOAL"),
                "goal_secondary": profile_data.get("SECONDARYGOAL"),
                "activity": profile_data.get("ACTIVITY"),
                "activity_level": profile_data.get("ACTIVITYLEVEL"),
                "mix_intended_use": profile_data.get("INTENDEDUSE"),
                "mix_frequency": profile_data.get("FREQUENCY"),
                "mix_timing": profile_data.get("MIXTIMING"),
                "health_issues": profile_data.get("HEALTHCONCERNS"),
            }
        ]

        recommendations_dict = {
            "email": user_email,
            "survey_date": survey_date,
            "first_name": user_name,
            "profile": profile_list,
            "recommended_protein_mix": recommendation["protein_mix"]
            # "customer_matches": [],
            # "sku_recommendations": []
        }

        recommendations = jsonable_encoder(recommendations_dict)

        store_in_cache(response_id, recommendations)

        return recommendations

    except HTTPException as e:
            raise e
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")



    # except Exception as e:
    #     logger.error(f"An unexpected error occurred for response_id: {response_id}: {traceback.format_exc()}"
    #     )
    #     log_error_in_snowflake(response_id, user_email, 500, f"An unexpected error occurred: {e}")
    #     raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

        

   
@app.get("/")
async def root():
    return {"message": "Get Recommendations"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."}
    )