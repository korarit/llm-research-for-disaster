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
model_id = "deepseek/deepseek-v4-flash"
model_name = "deepseek-v4-flash"

stage1_tools = [
    {
        "type": "function",
        "function": {
            "name": "filter_help_request",
            "description": "Determine if the disaster-related post is a direct request for help/rescue/immediate supplies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_help_request": {
                        "type": "boolean",
                        "description": "True if the message is a direct request for emergency rescue, medical aid, or immediate basic needs (help_request). False if it is a general update, weather warning, prayer, general donation campaign, or other non-emergency content (other)."
                    }
                },
                "required": ["is_help_request"]
            }
        }
    }
]

contact_detail_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": ["string", "null"],
            "description": "Full name, first name, prefix + name, or nickname of the person. Set to null if not mentioned."
        },
        "nickname": {
            "type": ["string", "null"],
            "description": "Nickname of the contact person (e.g., แบงค์, ส้ม, ป้าดา) if explicitly mentioned. Set to null if not mentioned."
        },
        "phone": {
            "type": ["string", "null"],
            "description": "Thai mobile phone number found for this person (e.g. 0812345678, 089-123-4567). Keep spelling exactly as written. Set to null if not mentioned."
        },
        "gender": {
            "type": ["string", "null"],
            "description": "Gender of the contact ('male' or 'female') inferred from prefix, nicknames, pronouns, or name. Set to null if cannot be determined."
        }
    },
    "required": ["name", "nickname", "phone", "gender"]
}

victims_count_schema = {
    "type": "object",
    "properties": {
        "dead": {
            "type": "integer",
            "description": "Number of dead people explicitly reported. Default to 0."
        },
        "critical": {
            "type": "integer",
            "description": "Number of people trapped, missing, in severe danger (e.g., RED triage: trapped on roof, landslide, swept away, unconscious, near-drowning, severe bleeding). Default to 0."
        },
        "urgent": {
            "type": "integer",
            "description": "Number of injured or sick people needing prompt assistance (e.g., YELLOW triage: bone fracture, high fever, severe diarrhea/vomiting, breathing difficulty). Default to 0."
        },
        "safe": {
            "type": "integer",
            "description": "Number of people reported safe/evacuated (e.g., GREEN triage: safe, evacuated, minor scratches). Default to 0."
        },
        "child": {
            "type": "integer",
            "description": "Number of children affected (age <= 11 or referred to as child/kid/น้อง/เด็ก/ทารก). Default to 0."
        },
        "bedridden": {
            "type": "integer",
            "description": "Number of bedridden patients (ผู้ป่วยติดเตียง, นอนติดเตียง, ป่วยติดเตียง). Default to 0."
        }
    },
    "required": ["dead", "critical", "urgent", "safe", "child", "bedridden"]
}

items_count_schema = {
    "type": "object",
    "properties": {
        "firstAid": {
            "type": "integer",
            "description": "Quantity/Need of first-aid kits or medicine (ยารักษาโรค, ยา, ชุดปฐมพยาบาล). Set to quantity needed, or 1 if needed but quantity is not specified. Default to 0."
        },
        "food": {
            "type": "integer",
            "description": "Quantity/Need of food/drinking water (อาหาร, น้ำดื่ม, ข้าวกล่อง, ของกิน). Set to quantity needed, or 1 if needed but quantity is not specified. Default to 0."
        },
        "energy": {
            "type": "integer",
            "description": "Quantity/Need of flashlights, powerbanks, candles, or backup power (ไฟสำรอง, แบตสำรอง, พาวเวอร์แบงค์, ไฟฉาย, เทียน, เครื่องปั่นไฟ). Set to quantity needed, or 1 if needed but quantity is not specified. Default to 0."
        }
    },
    "required": ["firstAid", "food", "energy"]
}

coordinates_detail_schema = {
    "type": "object",
    "properties": {
        "location_name": {
            "type": ["string", "null"],
            "description": "Specific location name, landmark, road, or sub-district name mentioned in the text. Set to null if not mentioned."
        },
        "google_map_url": {
            "type": ["string", "null"],
            "description": "Google Map URL (e.g., https://maps.app.goo.gl/...) found in the text. Set to null if not mentioned."
        },
        "lat": {
            "type": "number",
            "description": "Latitude coordinate (e.g., 13.7563) found in the text. Set to 0.0 if not mentioned."
        },
        "lng": {
            "type": "number",
            "description": "Longitude coordinate (e.g., 100.5018) found in the text. Set to 0.0 if not mentioned."
        }
    },
    "required": ["location_name", "google_map_url", "lat", "lng"]
}

ner_result_schema = {
    "type": "object",
    "properties": {
        "message_more_detail": {
            "type": "string",
            "description": "Brief summary of the disaster incident details in Thai"
        },
        "contact_victim": {
            "anyOf": [
                contact_detail_schema,
                {"type": "null"}
            ],
            "description": "Contact details of the victim. Set to null if not mentioned. If the victim is reporting for themselves (first-person), this should contain their details."
        },
        "contact_reporter": {
            "anyOf": [
                contact_detail_schema,
                {"type": "null"}
            ],
            "description": "Contact details of the reporter/informant who is reporting on behalf of the victim. Set to null if not mentioned. If first-person report, this should be the same as contact_victim."
        },
        "victims": victims_count_schema,
        "items": items_count_schema,
        "coordinates": coordinates_detail_schema
    },
    "required": [
        "message_more_detail",
        "contact_victim",
        "contact_reporter",
        "victims",
        "items",
        "coordinates"
    ]
}

agent2_tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_information",
            "description": "Extract named entities, contact information, victim counts, needed items, and coordinates from disaster messages.",
            "parameters": ner_result_schema
        }
    }
]

def parse_stage2_args(args):
    result = {
        "pred_message_more_detail": "",
        "pred_victim_name": None,
        "pred_victim_nickname": None,
        "pred_victim_phone": None,
        "pred_victim_gender": None,
        "pred_reporter_name": None,
        "pred_reporter_nickname": None,
        "pred_reporter_phone": None,
        "pred_reporter_gender": None,
        "pred_victims_dead": 0,
        "pred_victims_critical": 0,
        "pred_victims_urgent": 0,
        "pred_victims_safe": 0,
        "pred_victims_child": 0,
        "pred_victims_bedridden": 0,
        "pred_items_firstaid": 0,
        "pred_items_food": 0,
        "pred_items_energy": 0,
        "pred_location": None,
        "pred_google_map_url": None,
        "pred_lat": 0.0,
        "pred_lng": 0.0
    }
    
    if not isinstance(args, dict):
        return result
        
    result["pred_message_more_detail"] = args.get("message_more_detail", "")
    
    # contact_victim
    cv = args.get("contact_victim")
    if isinstance(cv, dict):
        result["pred_victim_name"] = cv.get("name")
        result["pred_victim_nickname"] = cv.get("nickname")
        result["pred_victim_phone"] = cv.get("phone")
        result["pred_victim_gender"] = cv.get("gender")
        
    # contact_reporter
    cr = args.get("contact_reporter")
    if isinstance(cr, dict):
        result["pred_reporter_name"] = cr.get("name")
        result["pred_reporter_nickname"] = cr.get("nickname")
        result["pred_reporter_phone"] = cr.get("phone")
        result["pred_reporter_gender"] = cr.get("gender")
        
    # victims
    v = args.get("victims")
    if isinstance(v, dict):
        result["pred_victims_dead"] = v.get("dead", 0)
        result["pred_victims_critical"] = v.get("critical", 0)
        result["pred_victims_urgent"] = v.get("urgent", 0)
        result["pred_victims_safe"] = v.get("safe", 0)
        result["pred_victims_child"] = v.get("child", 0)
        result["pred_victims_bedridden"] = v.get("bedridden", 0)
        
    # items
    it = args.get("items")
    if isinstance(it, dict):
        result["pred_items_firstaid"] = it.get("firstAid", 0)
        result["pred_items_food"] = it.get("food", 0)
        result["pred_items_energy"] = it.get("energy", 0)
        
    # coordinates
    c = args.get("coordinates")
    if isinstance(c, dict):
        result["pred_location"] = c.get("location_name")
        result["pred_google_map_url"] = c.get("google_map_url")
        result["pred_lat"] = c.get("lat", 0.0)
        result["pred_lng"] = c.get("lng", 0.0)
        
    return result

def predict_stage1(text, temp, retries=5):
    system_prompt = "You are an expert disaster response analyst. Your task is to analyze social media posts (tweets or Facebook comments in Thai) and determine if they contain a direct request for emergency rescue, medical aid, or immediate assistance (such as food/water/power) for specific victims in danger."
    user_prompt = f"""Analyze the following post and determine if it is a direct help request:

Post: "{text}"

CLASSIFICATION RULES:
- set is_help_request to True (help_request) if the post contains an active report of victims needing rescue, medical aid, or immediate basic supplies (food, clean water, first aid, power).
- set is_help_request to False (other) if the post is:
  - A general weather warning, rain warning, or evacuation announcement from authorities without reporting active trapped/injured victims.
  - A post of moral support, prayers, or expressing condolences (e.g., "ขอให้ปลอดภัย", "ส่งกำลังใจให้").
  - A general donation campaign, relief supply collection, or volunteer recruitment (e.g., "เปิดรับบริจาค", "รับบริจาค").
  - A general situational update on water levels, road status, or weather without active victim rescue reports.

Call the function 'filter_help_request' with your decision."""

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
                tools=stage1_tools,
                tool_choice={"type": "function", "function": {"name": "filter_help_request"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                is_help = args.get("is_help_request", False)
                if isinstance(is_help, str):
                    is_help = is_help.lower() == "true"
                return is_help, prompt_tokens, completion_tokens, latency
            return False, prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Stage 1 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return False, 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0
    return False, 0, 0, 0.0

def predict_stage2(text, temp, retries=5):
    system_prompt = "You are an expert disaster response information analyst. Your task is to analyze Thai social media posts about flood disasters and extract key named entities, contact information, victim counts, needed items, and coordinates from the text."
    user_prompt = f"""Analyze the following post and extract information according to the definitions and rules:

Post: "{text}"

EXTRACTION RULES:

1. CONTACT DETAILS (contact_victim and contact_reporter):
   - Identify if the post is a first-person report (the victim reports for themselves, e.g., using "ผม", "ฉัน", "หนู" to describe their own situation) or a third-person report (a reporter reports on behalf of a victim).
   - contact_victim: The person who is in danger/needs help. If it is a first-person report, extract their name, nickname, phone, and gender here. If third-person, extract the victim's details here.
   - contact_reporter: The person reporting the incident. If it is a first-person report, this should contain the exact same details as contact_victim. If third-person, extract the reporter's details here.
   - For both contacts, extract:
     - name: Full name (including prefix like นาย, นาง, คุณ, พี่, น้อง, เจ๊, เฮีย, ลุง, ป้า, ยาย, ตา, หมอ) if mentioned. If only a nickname is used as their name, put it in 'name'. Set to null if not mentioned.
     - nickname: Extract the nickname (e.g., แบงค์, ส้ม, ป้าดา) if explicitly mentioned. Set to null if not mentioned.
     - phone: Extract the Thai mobile phone number (e.g., starts with 08, 09, 06). Keep it exactly as written in the text (with dashes, spaces, or raw digits). Set to null if not mentioned.
     - gender: Infer gender ('male' or 'female') from prefixes, pronouns (ผม/ครับ -> male, ค่ะ/หนู/ฉัน -> female), nicknames, or typical Thai names. Set to null if cannot be determined.

2. VICTIMS COUNT (victims):
   - Extract counts of affected individuals based on their situation/symptom details in the text:
     - dead: number of deceased/dead individuals explicitly mentioned.
     - critical: number of victims in critical danger or RED triage condition (e.g., trapped on roof, landslide/debris collapse, swept away, unconscious/unresponsive, near-drowning, severe bleeding).
     - urgent: number of victims injured or sick needing prompt help or YELLOW triage condition (e.g., bone fracture, high fever, severe diarrhea/vomiting, breathing difficulty).
     - safe: number of survivors confirmed safe or evacuated, or GREEN triage (e.g., minor scratches, evacuated but safe).
     - child: number of children affected (age <= 11, or described as "เด็กเล็ก", "ลูกสาวคนเล็ก", "น้อง", "ทารก").
     - bedridden: number of bedridden patients affected (ผู้ป่วยติดเตียง, ป่วยติดเตียง, นอนติดเตียง).
   - If any count is not explicitly specified, set to 0. Do not guess counts if not mentioned in the text.

3. ITEMS NEEDED (items):
   - Extract quantities of relief items needed. Set to the exact quantity if mentioned. If an item is needed but no quantity is specified, set to 1. If not needed, set to 0.
     - firstAid: first-aid kits, medicine, medical supplies (ยารักษาโรค, ยา, ชุดปฐมพยาบาล).
     - food: food, drinking water, meal boxes, food supplies (อาหาร, น้ำดื่ม, ข้าวกล่อง, ของกิน).
     - energy: backup power, powerbanks, generators, flashlights, candles (ไฟสำรอง, แบตสำรอง, พาวเวอร์แบงค์, ไฟฉาย, เทียน, เครื่องปั่นไฟ).

4. COORDINATES & MAPS (coordinates):
   - location_name: The exact location name, landmark, road, village, or sub-district mentioned in the text. Keep the name exactly as written. Set to null if no location is mentioned.
   - google_map_url: The Google Maps URL (e.g., https://maps.app.goo.gl/...) found in the text. Set to null if not present.
   - lat & lng: Extract the latitude and longitude float values (e.g., "13.7563", "100.5018") if explicitly written as numbers in the text. Set both to 0.0 if not present. Do not look up or geocode coordinates.

Call the function 'extract_information' with the extracted details."""

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
                tool_choice={"type": "function", "function": {"name": "extract_information"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                return parse_stage2_args(args), prompt_tokens, completion_tokens, latency
            return parse_stage2_args({}), prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Stage 2 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return parse_stage2_args({}), 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0
    return parse_stage2_args({}), 0, 0, 0.0

def process_row(row, temp):
    synthetic_id = row['synthetic_id']
    generated_text = row['generated_text']
    
    is_help, s1_in, s1_out, s1_lat = predict_stage1(generated_text, temp)
    
    if is_help:
        stage2_data, s2_in, s2_out, s2_lat = predict_stage2(generated_text, temp)
    else:
        stage2_data = parse_stage2_args({})
        s2_in, s2_out, s2_lat = 0, 0, 0.0
        
    res = {
        "synthetic_id": synthetic_id,
        "generated_text": generated_text,
        "gt_is_help_request": row.get('gt_is_help_request'),
        "gt_classification_category": row.get('gt_classification_category'),
        "gt_location_name": row.get('gt_location_name'),
        "gt_google_map_url": row.get('gt_google_map_url'),
        "gt_lat": row.get('gt_lat'),
        "gt_lng": row.get('gt_lng'),
        "gt_victim_name": row.get('gt_victim_name'),
        "gt_victim_nickname": row.get('gt_victim_nickname'),
        "gt_victim_phone": row.get('gt_victim_phone'),
        "gt_victim_gender": row.get('gt_victim_gender'),
        "gt_reporter_name": row.get('gt_reporter_name'),
        "gt_reporter_nickname": row.get('gt_reporter_nickname'),
        "gt_reporter_phone": row.get('gt_reporter_phone'),
        "gt_reporter_gender": row.get('gt_reporter_gender'),
        "gt_dead": row.get('gt_dead'),
        "gt_critical": row.get('gt_critical'),
        "gt_urgent": row.get('gt_urgent'),
        "gt_safe": row.get('gt_safe'),
        "gt_child": row.get('gt_child'),
        "gt_bedridden": row.get('gt_bedridden'),
        "gt_item_firstaid": row.get('gt_item_firstaid'),
        "gt_item_food": row.get('gt_item_food'),
        "gt_item_energy": row.get('gt_item_energy'),
        
        "pred_is_help_request": is_help,
        "stage1_latency_seconds": s1_lat,
        "stage2_latency_seconds": s2_lat,
        "latency_seconds": s1_lat + s2_lat,
        "token_in_use": s1_in + s2_in,
        "token_out_use": s1_out + s2_out,
        "temperature": temp
    }
    res.update(stage2_data)
    
    # Add exact columns required by 5.1
    res["tweet_id"] = synthetic_id
    res["translated_thai"] = generated_text
    res["true_text_info"] = "informative" if row.get('gt_is_help_request') else "not_informative"
    res["true_text_human"] = row.get('gt_classification_category')
    res["predicted_message_detail"] = res["pred_message_more_detail"]
    res["predicted_victims_dead"] = res["pred_victims_dead"]
    res["predicted_victims_critical"] = res["pred_victims_critical"]
    res["predicted_victims_urgent"] = res["pred_victims_urgent"]
    res["predicted_victims_safe"] = res["pred_victims_safe"]
    res["predicted_victims_child"] = res["pred_victims_child"]
    res["predicted_victims_bedridden"] = res["pred_victims_bedridden"]
    res["predicted_items_firstaid"] = res["pred_items_firstaid"]
    res["predicted_items_food"] = res["pred_items_food"]
    res["predicted_items_energy"] = res["pred_items_energy"]
    res["predicted_location"] = res["pred_location"]
    res["tweet_text_char_count"] = 0
    res["translated_thai_char_count"] = len(generated_text)
    
    return res

def main():
    df = pd.read_csv("e:/nlp-for-disaster/dataset/clean/synthetic_ner_dataset.csv")
    output_dir = "e:/nlp-for-disaster/exp4/results"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{model_name}_results.csv")
    
    temp = 0.2
    
    existing_ids = set()
    processed_results = []
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if not existing_df.empty:
                print(f"[{model_name}] Found existing results file with {len(existing_df)} rows. Resuming...")
                processed_results = existing_df.to_dict(orient='records')
                existing_ids = set(existing_df['synthetic_id'].tolist())
        except Exception as e:
            print(f"Error loading existing CSV: {e}. Will overwrite.")
            
    # Filter rows to run
    rows_to_process = [row for _, row in df.iterrows() if row['synthetic_id'] not in existing_ids]
    
    if not rows_to_process:
        print(f"[{model_name}] All {len(df)} rows already evaluated.")
        return
        
    print(f"[{model_name}] Remaining rows to process: {len(rows_to_process)}/{len(df)}")
    
    # Run ThreadPoolExecutor
    new_results = []
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_row, row, temp): row['synthetic_id'] for row in rows_to_process}
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    new_results.append(res)
                    completed += 1
                    if completed % 50 == 0:
                        print(f"[{model_name}] Completed: {completed}/{len(rows_to_process)}")
                        # Periodically save progress
                        temp_combined = processed_results + new_results
                        temp_combined_df = pd.DataFrame(temp_combined)
                        temp_combined_df = temp_combined_df.set_index('synthetic_id').loc[df['synthetic_id']].reset_index()
                        temp_combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                except Exception as ex:
                    sid = futures[future]
                    print(f"[{model_name}] Error processing {sid}: {ex}")
    except KeyboardInterrupt:
        print(f"[{model_name}] Interrupted. Saving current progress...")
        
    # Save final results
    all_results = processed_results + new_results
    if all_results:
        final_df = pd.DataFrame(all_results)
        # Order the results to match original dataset order
        final_df = final_df.set_index('synthetic_id').loc[df['synthetic_id']].reset_index()
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[{model_name}] Done. Results saved to {output_csv}")

if __name__ == '__main__':
    main()
