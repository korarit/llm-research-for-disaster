import os
import time
import json
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY")
)
model_id = "google/gemma-4-26b-a4b-it"
model_name = "gemma-4"

CATEGORIES = [
    "not_informative",
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage"
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify",
            "description": "Classify the disaster-related tweet into the single most dominant category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "The dominant category of the tweet."
                    }
                },
                "required": ["category"]
            }
        }
    }
]

def predict(tweet_text, temp, retries=5):
    system_prompt = """You are a humanitarian disaster information analyst. Your task is to classify social media posts (tweets) collected during disaster events into exactly one category for emergency response analysis."""

    user_prompt = f"""Tweet: "{tweet_text}"

Classify the tweet into exactly ONE of the following categories. Choose the category that best represents the primary subject of the tweet:

CATEGORY DEFINITIONS:
1. not_informative: Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.
2. injured_or_dead_people: Reports of casualties, deaths, fatalities, injuries, or hospitalized people. (Signal words: killed, dead, casualties, injured, hospitalized)
3. missing_or_found_people: Reports of specific individuals or groups who are missing, active searches, or confirmed rescues. (Signal words: missing, search for, found, rescued, unaccounted)
4. affected_individuals: Evacuees, displaced people, survivors, homeless, stranded, or those taking shelter (WITHOUT reported deaths or injuries). (Signal words: evacuated, displaced, homeless, shelter, stranded)
5. infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power grids, water lines, or utility outages. (Signal words: collapsed, damaged, outage, flooded, blackout)
6. vehicle_damage: Damage to cars, trucks, boats, trains, or planes as the primary subject. (Signal words: car submerged, vehicle damaged)
7. rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, rescue operations, or emergency helpline sharing. (Signal words: donate, volunteers, aid, rescue team, relief, funding)
8. other_relevant_information: General news, weather forecasts, warning alerts, magnitude reports, or opinions about the disaster that do not report specific human or physical impact. (Signal words: warning, forecast, category, magnitude, news, report)

CRITICAL DECISION HIERARCHY (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_informative (Only for completely irrelevant or generic sentiment posts with no disaster details)

EDGE-CASE RESOLUTION RULES:
- "Prayers for Nepal #earthquake" -> Classify as 'other_relevant_information' (contains specific disaster keyword).
- "Prayers for everyone" -> Classify as 'not_informative' (no specific disaster reference).
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

Return your classification by calling the 'classify' function."""

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
                tool_choice={"type": "function", "function": {"name": "classify"}},
                temperature=temp
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            choice = response.choices[0]
            if choice.message.tool_calls:
                arg_str = choice.message.tool_calls[0].function.arguments
                args = json.loads(arg_str)
                category = args.get("category", "not_informative")
                return category, prompt_tokens, completion_tokens, latency

            return "not_informative", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "not_informative", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    tweet_text = row['tweet_text']
    true_info = row['text_info']
    true_human = row['text_human']

    pred_category, in_tokens, out_tokens, latency = predict(tweet_text, temp)

    if pred_category == "not_informative":
        mapped_info = "not_informative"
        mapped_cat = "not_humanitarian"
    else:
        mapped_info = "informative"
        mapped_cat = pred_category

    return {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "predicted_category": pred_category,
        "mapped_predicted_info": mapped_info,
        "mapped_predicted_category": mapped_cat,
        "tweet_text_char_count": len(tweet_text),
        "token_in_use": in_tokens,
        "token_out_use": out_tokens,
        "latency_seconds": latency
    }

def main():
    df = pd.read_csv("e:/nlp-for-disaster/dataset/dataset_sample_500.csv")
    results = []

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
    output_dir = "e:/nlp-for-disaster/exp1E/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
