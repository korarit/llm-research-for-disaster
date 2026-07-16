import sys
import os

# Ensure we import client from full_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from full_agent.agent2_ner import extract_ner

# A test tweet containing nested quotes or complex characters that could trigger unescaped quote issues
tweet = "ช่วยด้วยค่ะ! บ้านป้าศรี \"สมรัก\" มีคนเจ็บติดอยู่ข้างใน 2 คน น้ำท่วมสูงเกือบถึงหลังคาแล้ว โทร 081-234-5678 พิกัด วัดบ้านกอก ต.ในเมือง"

print("Calling extract_ner with deepseek-v4-flash...")
result, prompt_tok, comp_tok, lat = extract_ner(tweet, "deepseek-v4-flash")

print("\n--- Results ---")
print("Prompt tokens:", prompt_tok)
print("Completion tokens:", comp_tok)
print("Latency:", lat)
print("Parsed JSON:")
import json
print(json.dumps(result, indent=2, ensure_ascii=False))
