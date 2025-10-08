# handler.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import torch
import runpod
from dotenv import load_dotenv
from RS_utils import CSACActor, encode_state_features, format_recommendation, get_or_refresh_connection, custom_serializer, extract_responses, convert_height_to_inches
import json
from datetime import datetime
import traceback
import pandas as pd
import requests
from pydantic_settings import BaseSettings
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("/app/logs/application.log", maxBytes=5*1024*1024, backupCount=7),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

actor = None
variant_ids = None
categories = None
redis_client = None

# Load env for Redis
import redis
redis_client = redis.StrictRedis(host=os.getenv('REDIS_HOST', '127.0.0.1'), port=6379, db=0)

def get_from_cache(response_id):
    try:
        cached_data = redis_client.get(response_id)
        if cached_data:
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None
    

def store_in_cache(response_id, data):
    try:
        redis_client.set(response_id, json.dumps(data, default=custom_serializer))
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

try:
    conn = get_or_refresh_connection()
    cursor = conn.cursor()
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
    variant_df = cursor.fetch_pandas_all()
    variant_ids = [str(v) for v in variant_df['VARIANTID'].tolist()]
    categories = [str(c) for c in variant_df['CATEGORY'].tolist()]
    logger.info("Variant data loaded successfully")
    
    state_dim = 120 
    action_dim = len(variant_ids)
    actor = CSACActor(state_dim=state_dim, action_dim=action_dim)
    with open('/app/models/improved_csac_actor.pth', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        actor.load_state_dict(state_dict)
    actor.eval()
    logger.info(f"Actor model loaded successfully with state_dim={state_dim}, action_dim={action_dim}")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
    raise Exception(f"Failed to initialize: {str(e)}")

AUTHORIZATION_TOKEN = settings.authorization_token
FORM_ID = settings.form_id
TYPEFORM_API_TOKEN = settings.typeform_api_token

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.credentials != AUTHORIZATION_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing token",
#         )
    

def handler(job):
    try:
        start_time = time.time()
        job_input = job.get("input", {})
        # email = job_input.get("email")
        response_id = job_input.get("response_id")
        if not response_id:
            logger.error("Missing response_id in input")
            return {"error": "Missing response_id"}
        
        logger.info(f"Fetching recommendations for response_id: {response_id}")

        # Check cache first
        recommendations = get_from_cache(response_id)
        if recommendations:
            logger.info(f"Cache hit for response_id: {response_id}")
            total_time = time.time() - start_time
            recommendations["total_time_taken"] = total_time
            return recommendations
        

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
            return {"error": "No responses found in Typeform"}
            
        typeform_item = typeform_data["items"][0]
        survey_date = typeform_item['submitted_at']
        user_name = None
        user_email = None
        for answer in typeform_item['answers']:
            field_ref = answer['field']['ref']
            if field_ref == settings.field_ref_firstname:
                user_name = answer.get('text', None)
            if field_ref == settings.field_ref_email:
                user_email = answer.get('email', None)

        user_df = extract_responses(typeform_item)
        state = encode_state_features(user_df)

        if state is None:
            return {"error": "Failed to encode state features"}
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action = [float(x) for x in action]
        recommendation = format_recommendation(action, variant_ids, categories)


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

        recommendations_json = json.dumps(recommendations_dict, default=custom_serializer)
        
        # Store in cache
        store_in_cache(response_id, recommendations_dict)

        # Snowflake logging
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.LOG_COACH_RS
            (ResponseId, ResponseJSON, Email, StatusCode, StatusMessage)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (response_id, recommendations_json, user_email, 200, "success")
        )
        
        cursor.execute("SELECT MAX(ID) FROM INGEST_DATA.TRUENUTRITION_BUILD.LOG_COACH_RS")
        log_id = cursor.fetchone()[0]
        
        for item in recommendation['protein_mix']:
            cursor.execute(
                """
                INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BASE_RESULT
                (LOG_ID, RESPONSEID, CATEGORY, ITEMNAME, VARIANTID, PERCENTOFMIX)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (log_id, response_id, item['category'], item['item_name'], item['variant_id'], item['percent'])
            )
        conn.commit()
        
        total_time = time.time() - start_time
        return {
            "response_id": response_id,
            "email": user_email,
            "recommendations": recommendation,
            "timestamp": datetime.utcnow().isoformat(),
            "total_time_taken": total_time
        }
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

runpod.serverless.start({"handler": handler})

# job_input = {
#     "input": {
#         # "email": "firedudepete@gmail.com",
#         "response_id": "t8rmpia65cg3ywqpbl5vlut8rmpiaepk"
#     }
# }

# if __name__ == "__main__":
#     output = handler(job_input)
#     print("Handler Output:")
#     print(output)