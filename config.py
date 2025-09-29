# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    snowflake_user: str
    snowflake_password: str
    snowflake_account: str
    snowflake_warehouse: str
    snowflake_database: str
    snowflake_schema: str
    snowflake_mfa_secret: str
    snowflake_ssh_pass: str
    typeform_api_token: str
    authorization_token: str

    field_ref_email: str
    field_ref_gender: str
    field_ref_age: str
    field_ref_height_inches: str
    field_ref_weight: str
    field_ref_body_type: str
    field_ref_body_fat: str
    field_ref_diet: str
    field_ref_diet_restrictions: str
    field_ref_macros: str
    field_ref_goal_primary: str
    field_ref_goal_secondary: str
    field_ref_activity: str
    field_ref_activity_level: str
    field_ref_mix_intended_use: str
    field_ref_mix_frequency: str
    field_ref_mix_timing: str
    field_ref_health_issues: str
    field_ref_firstname: str
    form_id: str
    
    reference_date: str
    top_k: int
    threshold: float
    
    k_months: int
    
    scheduler_hour: int
    scheduler_minute: int
    scheduler_timezone: str
    
    logging_flag: bool
    
    recommend_beyond_constraints_flag: bool

    class Config:
        env_file = ".env"

settings = Settings()
