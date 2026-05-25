import json
import torch
import pandas as pd
from modeling.train_transformer import TransformerModel
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

MODEL_ID = "gemini-2.5-flash"

def call_gemini(prompt: str) -> str:
    """
    Calls Gemini using the new Google GenAI SDK.
    Uses client.models.generate_content with a dict config.
    """
    client = genai.Client(api_key=API_KEY)

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        # defensive extraction in case shape changes
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        # fallback if structure is different
        return str(response)

    except Exception as e:
        print("Gemini SDK error:", type(e).__name__)
        print("Details:", str(e))
        # Do NOT touch `response` here; it might not exist
        raise

model = TransformerModel(input_dim=14)
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

with open('synthetic_maintenance_records.json', 'r') as f:
    records = json.load(f)

def get_logs(engine_id):
    return [r['description'] for r in records if r['engine_id'] == engine_id]

def llm_enhance(sensor_seq, engine_id, transformer_anom, transformer_rul):
    logs_str = ' '.join(get_logs(engine_id)[:5])  # Limit for prompt size
    sensor_summary = f"Anomaly prob: {transformer_anom}, Est RUL: {transformer_rul}"

    prompt = f"""
You are a predictive maintenance expert. 
Given sensor summary: {sensor_summary}.
Historical logs: {logs_str}.
Refine the prediction: Output failure mode and adjusted RUL as JSON 
{{'failure_mode': 'str', 'rul': int}}.
"""

    response_text = call_gemini(prompt)
    return response_text  
df_test = pd.read_csv('test_sensors.csv')

sensor_cols = [
    'setting_1', 'setting_2',
    'T2_total_temp', 'T24_total_temp', 'T30_total_temp', 'T50_total_temp',
    'P15_static_pressure', 'P30_total_pressure',
    'nf_fan_speed', 'nc_core_speed', 'epr_engine_pressure_ratio',
    'ps30_static_pressure', 'farb_fuel_air_ratio_burner',
    'vibration'
]

# Take first sequence (50 timesteps)
sample_seq = df_test.iloc[0:50][sensor_cols].values
sample_seq = torch.tensor(sample_seq, dtype=torch.float32).unsqueeze(0)  # Batch dim

# Transformer predictions
anom_pred, rul_pred = model(sample_seq)
anom_prob = torch.softmax(anom_pred, dim=1)[0][1].item()  # Anomaly probability
rul_est = rul_pred.item()

# LLM enhancement
enhanced = llm_enhance(sample_seq, df_test.iloc[0]['engine_id'], anom_prob, rul_est)
print(enhanced)