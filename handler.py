# handler.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import torch
import runpod
from dotenv import load_dotenv
from RS_utils import CSACActor, encode_state_features, format_recommendation, get_or_refresh_connection, custom_serializer
import json
from datetime import datetime
import traceback
import pandas as pd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("logs/application.log", maxBytes=5*1024*1024, backupCount=7),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

actor = None
variant_ids = None
categories = None

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
    with open('Improved_csac_actor.pth', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        actor.load_state_dict(state_dict)
    actor.eval()
    logger.info(f"Actor model loaded successfully with state_dim={state_dim}, action_dim={action_dim}")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
    raise Exception(f"Failed to initialize: {str(e)}")

def handler(job):
    try:
        start_time = time.time()
        job_input = job.get("input", {})
        email = job_input.get("email")
        response_id = job_input.get("response_id")
        if not email or not response_id:
            logger.error("Missing email or response_id in input")
            return {"error": "Missing email or response_id"}
        
        state = encode_state_features(email, response_id)
        if state is None:
            return {"error": "Failed to encode state features"}
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action = [float(x) for x in action]
        
        recommendation = format_recommendation(action, variant_ids, categories)
        for item in recommendation.get('protein_mix', []):
            logger.info(f"Protein mix item: {item}, type: {type(item)}")
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        recommendations_json = json.dumps({
            "email": str(email),
            "response_id": str(response_id),
            "recommendations": recommendation,
            "timestamp": datetime.utcnow().isoformat()
        }, default=custom_serializer)
        
        cursor.execute(
            """
            INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.LOG_COACH_RS
            (ResponseId, ResponseJSON, Email, StatusCode, StatusMessage)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (response_id, recommendations_json, email, 200, "success")
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
            "email": email,
            "recommendations": recommendation,
            "timestamp": datetime.utcnow().isoformat(),
            "total_time_taken": total_time
        }
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

runpod.serverless.start({"handler": handler})

