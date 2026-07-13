import os
import time
import json
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Clients & Config
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY")
)
model_id = "google/gemma-4-26b-a4b-it"
model_name = "gemma-4"

agent1_tools = [
    {
        "type": "function",
        "function": {
            "name": "filter_informativeness",
            "description": "Determine if the tweet contains SPECIFIC disaster impact/response evidence, facts, or details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "informativeness": {
                        "type": "string",
                        "enum": ["informative", "not_informative"],
                        "description": "Whether the tweet is informative."
                    }
                },
                "required": ["informativeness"]
            }
        }
    }
]

agent2_tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_category",
            "description": "Classify the disaster-related tweet into the dominant category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "affected_individuals",
                            "infrastructure_and_utility_damage",
                            "injured_or_dead_people",
                            "missing_or_found_people",
                            "other_relevant_information",
                            "rescue_volunteering_or_donation_effort",
                            "vehicle_damage"
                        ],
                        "description": "Identify the dominant content category."
                    }
                },
                "required": ["category"]
            }
        }
    }
]

def predict_agent1(tweet_text, temp, retries=5):
    system_prompt = """You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.

BIAS RULE: When in doubt, lean toward 'informative'. Only classify as 'not_informative' when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case."""
    user_prompt = f"""Tweet: "{tweet_text}"

Does this tweet contain SPECIFIC, FACTUAL information about a disaster event?

informative — Mark as informative if the tweet contains ANY of:
  - Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  - Named locations with concrete disaster impact described
  - Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  - Reports of rescue, aid, or emergency response operations with details
  - Measurable data: wind speed, magnitude, flood levels, temperature extremes
  - Weather forecasts, warnings, storm tracks, magnitude reports, or direct discussion referencing a specific disaster (e.g., "ส่งกำลังใจให้ผู้ประสบภัยน้ำท่วมเชียงราย #น้ำท่วมเชียงราย" contains the Chiang Rai flood keyword).

not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  - Pure emotional expression (prayers, sympathy, fear, grief) with NO specific disaster references or details (e.g., "ขอส่งกำลังใจให้ทุกคนปลอดภัย").
  - Political argument or blame with NO specific disaster impact described.
  - Jokes, obvious sarcasm, or clear misinformation.
  - Completely off-topic content unrelated to the disaster.

Call the function 'filter_informativeness' with your decision."""

    backoff = 1.0
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=agent1_tools,
                tool_choice={"type": "function", "function": {"name": "filter_informativeness"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                informativeness = args.get("informativeness", "not_informative")
                return informativeness, prompt_tokens, completion_tokens, latency
                
            return "not_informative", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Agent 1 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "not_informative", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def predict_agent2(tweet_text, temp, retries=5):
    system_prompt = """You are a disaster content categorizer. The tweet you receive has already been confirmed as containing specific disaster information. Your job is to identify its PRIMARY humanitarian content category.

CORE PRINCIPLE: Choose the MOST SPECIFIC category that fits the tweet's dominant subject. Use 'other_relevant_information' only as a last resort when no other category applies."""
    user_prompt = f"""Tweet: "{tweet_text}"

This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category using this priority order:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Thai signal words: เสียชีวิต, ตาย, เสียชีวิตแล้ว, ผู้เสียชีวิต, พบร่าง, พบศพ, ยอดเสียชีวิต, บาดเจ็บ, ได้รับบาดเจ็บ, เจ็บ, เจ็บสาหัส, บาดเจ็บสาหัส, ส่งโรงพยาบาล, ส่งรพ., รักษาตัวที่โรงพยาบาล, กู้ชีพพบร่าง
   ⚠ Takes priority over affected_individuals and infrastructure damage if any death or injury is mentioned.

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Thai signal words: สูญหาย, หายตัว, หาย, สูญหายไป, ตามหา, ค้นหา, ค้นหาผู้สูญหาย, พบตัวแล้ว, เจอแล้ว, พบตัว, ช่วยชีวิตได้แล้ว, ช่วยเหลือได้แล้ว, รอดชีวิต, ปลอดภัยแล้ว, ติดต่อไม่ได้, ยังไม่พบตัว, ขาดการติดต่อ
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment.

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Thai signal words: อพยพ, ถูกอพยพ, หนีน้ำ, พลัดถิ่น, ไร้ที่อยู่อาศัย, ไร้บ้าน, ศูนย์อพยพ, สถานที่พักพิง, จุดพักพิง, ติดอยู่, ติดค้าง, ออกไม่ได้, ผู้รอดชีวิต, ผู้ประสบภัย, ชาวบ้านเดือดร้อน, บ้านน้ำท่วม
   ⚠ Only use when NO deaths or injuries are mentioned.

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Thai signal words: ถล่ม, ทรุดตัว, พังทลาย, ยุบตัว, พังเสียหาย, ได้รับความเสียหาย, เสียหาย, พัง, ไฟดับ, น้ำประปาไม่ไหล, ไม่มีไฟฟ้า, สัญญาณขาดหาย, น้ำท่วม, ถนนถูกน้ำท่วม, ท่วมถนน, น้ำท่วมขัง, ถนนขาด, ถนนพัง, สะพานขาด, สะพานพัง, เส้นทางชำรุด, เสาไฟล้ม, อาคารถล่ม
   ⚠ Do NOT use when vehicles are the primary subject.

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Thai signal words: รถจมน้ำ, รถยนต์จมน้ำ, รถพัง, รถยนต์เสียหาย, รถได้รับความเสียหาย, เรือล่ม, เรือพัง, เรืออับปาง, เรือจม, รถไหลไปกับน้ำ, รถโดนพัดไป, รถคว่ำ, รถบรรทุกคว่ำ
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention.

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization, emergency helpline sharing, relief distribution
   Thai signal words: บริจาค, เปิดรับบริจาค, เงินบริจาค, สมทบทุน, ระดมทุน, ร่วมบริจาค, จิตอาสา, อาสาสมัคร, อาสา, ถุงยังชีพ, ข้าวกล่อง, แจกของ, ความช่วยเหลือ, สิ่งของช่วยเหลือ, แจกจ่ายสิ่งของ, หน่วยกู้ภัย, ทีมกู้ภัย, กู้ภัย, กู้ชีพ, ลงพื้นที่ช่วยเหลือ
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons.

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, official warnings without impact details, general news/opinions/expressions of solidarity that mention a specific disaster.
   ⚠ If any specific category above fits, use that instead.

EDGE-CASE RESOLUTION RULES:
- "ศูนย์อพยพวัดศรีทรายมูลกำลังแจกข้าวกล่องและน้ำดื่ม" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "สะพานขาด รถสัญจรไม่ได้ที่แม่สาย" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

Call the function 'classify_category' with your decision."""

    backoff = 1.0
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=agent2_tools,
                tool_choice={"type": "function", "function": {"name": "classify_category"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                category = args.get("category", "other_relevant_information")
                return category, prompt_tokens, completion_tokens, latency
                
            return "other_relevant_information", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Agent 2 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "other_relevant_information", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    translated_thai = row['translated_thai']
    original_english = row['original_english']
    true_info = row['true_text_info']
    true_human = row['true_text_human']
    
    # Agent 1 Informativeness Filter
    pred_info, a1_in, a1_out, a1_lat = predict_agent1(translated_thai, temp)
    
    if pred_info == "informative":
        # Agent 2 Category Classifier
        pred_cat, a2_in, a2_out, a2_lat = predict_agent2(translated_thai, temp)
        final_info = "informative"
        final_cat = pred_cat
    else:
        pred_cat = ""
        a2_in, a2_out, a2_lat = 0, 0, 0.0
        final_info = "not_informative"
        final_cat = "not_humanitarian"
        
    return {
        "tweet_id": tweet_id,
        "translated_thai": translated_thai,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "agent1_predicted_info": pred_info,
        "agent2_predicted_category": pred_cat,
        "final_predicted_info": final_info,
        "final_predicted_category": final_cat,
        "tweet_text_char_count": len(str(original_english)) if pd.notna(original_english) else 0,
        "translated_thai_char_count": len(str(translated_thai)) if pd.notna(translated_thai) else 0,
        "token_in_use": a1_in + a2_in,
        "token_out_use": a1_out + a2_out,
        "agent1_latency_seconds": a1_lat,
        "agent2_latency_seconds": a2_lat,
        "latency_seconds": a1_lat + a2_lat
    }

def main():
    df = pd.read_csv("e:/nlp-for-disaster/dataset/CrisisMMD_Thai_1000.csv")
    results = []
    
    existing_0_csv = "e:/nlp-for-disaster/exp3th/results/gemma-4_results_th.csv"
    temps_to_run = [0.1, 0.2, 0.3]
    if os.path.exists(existing_0_csv):
        print(f"[{model_name}] Found existing 0.0 results, loading them...")
        df_0 = pd.read_csv(existing_0_csv)
        df_0['temperature'] = 0.0
        cols_to_keep = ["tweet_id", "translated_thai", "true_text_info", "true_text_human", "agent1_predicted_info", 
                        "agent2_predicted_category", "final_predicted_info", "final_predicted_category", 
                        "tweet_text_char_count", "translated_thai_char_count", "token_in_use", "token_out_use", 
                        "agent1_latency_seconds", "agent2_latency_seconds", "latency_seconds", "temperature"]
        df_0 = df_0[cols_to_keep]
        results.append(df_0)
    else:
        temps_to_run = [0.0, 0.1, 0.2, 0.3]
        
    for temp in temps_to_run:
        print(f"Starting {model_name} for temperature {temp}...")
        temp_results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_row, row, temp): idx for idx, row in df.iterrows()}
            completed = 0
            for future in as_completed(futures):
                res = future.result()
                res['temperature'] = temp
                temp_results.append(res)
                completed += 1
                if completed % 100 == 0:
                    print(f"[{model_name} Temp {temp}] Progress: {completed}/{len(df)}")
                    
        temp_df = pd.DataFrame(temp_results)
        temp_df = temp_df.set_index('tweet_id').loc[df['tweet_id']].reset_index()
        results.append(temp_df)
        
    final_df = pd.concat(results, ignore_index=True)
    output_dir = "e:/nlp-for-disaster/exp3th/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results_th.csv"), index=False)
    
    # Save the 0.0 temp results as baseline results too
    df_0_save = final_df[final_df['temperature'] == 0.0]
    df_0_save.to_csv(os.path.join(output_dir, f"{model_name}_results_th.csv"), index=False)
    
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
