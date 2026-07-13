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
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_API_KEY")
)
model_id = "typhoon-v2.5-30b-a3b-instruct"
model_name = "typhoon-v2.5"

CATEGORIES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage"
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_two_layer",
            "description": "Classify the tweet into informativeness and humanitarian category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "informativeness": {
                        "type": "string",
                        "enum": ["informative", "not_informative"],
                        "description": "determine if the tweet contains SPECIFIC disaster impact/response evidence, facts, or details"
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "identify the DOMINANT content (choose only ONE)"
                    }
                },
                "required": ["informativeness", "category"]
            }
        }
    }
]

def predict(tweet_text, temp, retries=5):
    system_prompt = "You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent."
    user_prompt = f"""Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or generic personal banter/sentiment (prayers/wishes) that does not refer to the specific disaster at all.

LAYER 2 — HUMANITARIAN CATEGORY
Identify the category that best represents the primary subject of the tweet:
- injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Thai signal words: เสียชีวิต, ตาย, เสียชีวิตแล้ว, ผู้เสียชีวิต, พบร่าง, พบศพ, ยอดเสียชีวิต, บาดเจ็บ, ได้รับบาดเจ็บ, เจ็บ, เจ็บสาหัส, บาดเจ็บสาหัส, ส่งโรงพยาบาล, ส่งรพ., รักษาตัวที่โรงพยาบาล, กู้ชีพพบร่าง)
- missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Thai signal words: สูญหาย, หายตัว, หาย, สูญหายไป, ตามหา, ค้นหา, ค้นหาผู้สูญหาย, พบตัวแล้ว, เจอแล้ว, พบตัว, ช่วยชีวิตได้แล้ว, ช่วยเหลือได้แล้ว, รอดชีวิต, ปลอดภัยแล้ว, ติดต่อไม่ได้, ยังไม่พบตัว, ขาดการติดต่อ)
- affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Thai signal words: อพยพ, ถูกอพยพ, หนีน้ำ, พลัดถิ่น, ไร้ที่อยู่อาศัย, ไร้บ้าน, ศูนย์อพยพ, สถานที่พักพิง, จุดพักพิง, ติดอยู่, ติดค้าง, ออกไม่ได้, ผู้รอดชีวิต, ผู้ประสบภัย, ชาวบ้านเดือดร้อน, บ้านน้ำท่วม)
- infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Thai signal words: ถล่ม, ทรุดตัว, พังทลาย, ยุบตัว, พังเสียหาย, ได้รับความเสียหาย, เสียหาย, พัง, ไฟดับ, น้ำประปาไม่ไหล, ไม่มีไฟฟ้า, สัญญาณขาดหาย, น้ำท่วม, ถนนถูกน้ำท่วม, ท่วมถนน, น้ำท่วมขัง, ถนนขาด, ถนนพัง, สะพานขาด, สะพานพัง, เส้นทางชำรุด, เสาไฟล้ม, อาคารถล่ม)
- vehicle_damage: Damage to cars, trucks, boats, or planes. (Thai signal words: รถจมน้ำ, รถยนต์จมน้ำ, รถพัง, รถยนต์เสียหาย, รถได้รับความเสียหาย, เรือล่ม, เรือพัง, เรืออับปาง, เรือจม, รถไหลไปกับน้ำ, รถโดนพัดไป, รถคว่ำ, รถบรรทุกคว่ำ)
- rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Thai signal words: บริจาค, เปิดรับบริจาค, เงินบริจาค, สมทบทุน, ระดมทุน, ร่วมบริจาค, จิตอาสา, อาสาสมัคร, อาสา, ถุงยังชีพ, ข้าวกล่อง, แจกของ, ความช่วยเหลือ, สิ่งของช่วยเหลือ, แจกจ่ายสิ่งของ, หน่วยกู้ภัย, ทีมกู้ภัย, กู้ภัย, กู้ชีพ, ลงพื้นที่ช่วยเหลือ)
- other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Thai signal words: เตือนภัย, ประกาศเตือน, เฝ้าระวัง, แจ้งเตือน, ประกาศจากราชการ, พยากรณ์อากาศ, คาดการณ์, ดินฟ้าอากาศ, ความรุนแรง, ริกเตอร์, ขนาดความแรง, ระดับน้ำ, ปริมาณน้ำฝน, ข่าวภัยพิบัติ, รายงานสถานการณ์, อัพเดทสถานการณ์, อัพเดทน้ำท่วม, ภาพดาวเทียม, เส้นทางพายุ, พายุเข้า)
- not_humanitarian: Use this ONLY if the tweet is classified as 'not_informative' in Layer 1.

CRITICAL DECISION HIERARCHY FOR LAYER 2 (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_humanitarian (Only when Layer 1 is 'not_informative')

CONSISTENCY RULE:
- If Layer 1 is 'not_informative', Layer 2 must be 'not_humanitarian'.
- If Layer 1 is 'informative', Layer 2 must NOT be 'not_humanitarian' (choose one of the other 7 categories instead).

EDGE-CASE RESOLUTION RULES:
- "ส่งกำลังใจให้ผู้ประสบภัยน้ำท่วมเชียงราย #น้ำท่วมเชียงราย" -> Layer 1: 'informative', Layer 2: 'other_relevant_information' (contains specific disaster keyword).
- "ขอส่งกำลังใจให้ทุกคนปลอดภัย" -> Layer 1: 'not_informative', Layer 2: 'not_humanitarian' (no specific disaster reference).
- "ศูนย์อพยพวัดศรีทรายมูลกำลังแจกข้าวกล่องและน้ำดื่ม" -> Layer 1: 'informative', Layer 2: 'rescue_volunteering_or_donation_effort'.
- "สะพานขาด รถสัญจรไม่ได้ที่แม่สาย" -> Layer 1: 'informative', Layer 2: 'infrastructure_and_utility_damage'.

Return classification by calling the function 'classify_two_layer' with both values."""

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
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "classify_two_layer"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                info = args.get("informativeness", "not_informative")
                cat = args.get("category", "not_humanitarian")
                return info, cat, prompt_tokens, completion_tokens, latency
                
            return "not_informative", "not_humanitarian", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "not_informative", "not_humanitarian", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    translated_thai = row['translated_thai']
    original_english = row['original_english']
    true_info = row['true_text_info']
    true_human = row['true_text_human']
    
    pred_info, pred_cat, in_tokens, out_tokens, latency = predict(translated_thai, temp)
    
    # Apply baseline mapping rule: if predicted_info is not_informative, force category to not_humanitarian
    if pred_info == "not_informative":
        mapped_cat = "not_humanitarian"
    else:
        mapped_cat = pred_cat
        
    return {
        "tweet_id": tweet_id,
        "translated_thai": translated_thai,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "predicted_info": pred_info,
        "predicted_category": mapped_cat,
        "tweet_text_char_count": len(str(original_english)) if pd.notna(original_english) else 0,
        "translated_thai_char_count": len(str(translated_thai)) if pd.notna(translated_thai) else 0,
        "token_in_use": in_tokens,
        "token_out_use": out_tokens,
        "latency_seconds": latency
    }

def main():
    df = pd.read_csv("e:/nlp-for-disaster/dataset/CrisisMMD_Thai_1000.csv")
    results = []
    
    existing_0_csv = "e:/nlp-for-disaster/exp2th/results/typhoon-v2.5_results_th.csv"
    temps_to_run = [0.1, 0.2, 0.3]
    if os.path.exists(existing_0_csv):
        print(f"[{model_name}] Found existing 0.0 results, loading them...")
        df_0 = pd.read_csv(existing_0_csv)
        df_0['temperature'] = 0.0
        cols_to_keep = ["tweet_id", "translated_thai", "true_text_info", "true_text_human", "predicted_info", 
                        "predicted_category", "tweet_text_char_count", "translated_thai_char_count", 
                        "token_in_use", "token_out_use", "latency_seconds", "temperature"]
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
    output_dir = "e:/nlp-for-disaster/exp2th/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results_th.csv"), index=False)
    
    # Save the 0.0 temp results as baseline results too
    df_0_save = final_df[final_df['temperature'] == 0.0]
    df_0_save.to_csv(os.path.join(output_dir, f"{model_name}_results_th.csv"), index=False)
    
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
