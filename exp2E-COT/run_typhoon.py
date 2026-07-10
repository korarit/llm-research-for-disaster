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
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_API_KEY")
)
model_id = "typhoon-v2.5-30b-a3b-instruct"
model_name = "typhoon-v2.5"

CATEGORIES = [
    "injured_or_dead_people",
    "missing_or_found_people",
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "vehicle_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian"
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_two_layer",
            "description": "Perform a two-layer classification with a short analysis of the tweet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_reasoning": {
                        "type": "string",
                        "description": "Analyze the tweet text for key clues (e.g. casualties, evacuation, collapsed) and explain in 1-2 sentences to justify both Layer 1 (informativeness) and Layer 2 (category) decisions."
                    },
                    "informativeness": {
                        "type": "string",
                        "enum": ["informative", "not_informative"],
                        "description": "Layer 1: Does the tweet contain any disaster-related information?"
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "Layer 2: The dominant humanitarian category representing the tweet."
                    }
                },
                "required": ["short_reasoning", "informativeness", "category"]
            }
        }
    }
]

def predict(tweet_text, temp, retries=5):
    system_prompt = """You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent."""

    user_prompt = f"""Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously. You must provide a brief analysis first:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or generic personal banter/sentiment (prayers/wishes) that does not refer to the specific disaster at all.

LAYER 2 — HUMANITARIAN CATEGORY
Identify the category that best represents the primary subject of the tweet:
- injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Signal words: killed, dead, casualties, injured, hospitalized)
- missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Signal words: missing, search for, found, rescued)
- affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Signal words: evacuated, displaced, homeless, shelter, stranded)
- infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Signal words: collapsed, damaged, outage, flooded, blackout)
- vehicle_damage: Damage to cars, trucks, boats, or planes. (Signal words: car submerged, vehicle damaged)
- rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Signal words: donate, volunteers, aid, rescue team, relief)
- other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Signal words: warning, forecast, category, magnitude, news, report)
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
- "Prayers for Nepal #earthquake" -> Layer 1: 'informative', Layer 2: 'other_relevant_information' (contains specific disaster keyword).
- "Prayers for everyone" -> Layer 1: 'not_informative', Layer 2: 'not_humanitarian' (no specific disaster reference).
- "Evacuees are being given food at the shelter" -> Layer 1: 'informative', Layer 2: 'rescue_volunteering_or_donation_effort'.
- "Bridge collapsed, blocking cars" -> Layer 1: 'informative', Layer 2: 'infrastructure_and_utility_damage'.

STEPS FOR WORKFLOW:
1. In the "short_reasoning" field, note down the critical clues and explain why the tweet belongs to your chosen categories in 1-2 sentences.
2. Call 'classify_two_layer' with your analysis and decisions."""

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
                reasoning = args.get("short_reasoning", "")
                info = args.get("informativeness", "not_informative")
                category = args.get("category", "not_humanitarian")
                return reasoning, info, category, prompt_tokens, completion_tokens, latency

            return "", "not_informative", "not_humanitarian", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "", "not_informative", "not_humanitarian", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    tweet_text = row['tweet_text']
    true_info = row['text_info']
    true_human = row['text_human']

    pred_reasoning, pred_info, pred_category, in_tokens, out_tokens, latency = predict(tweet_text, temp)

    if pred_info == "not_informative":
        mapped_cat = "not_humanitarian"
    else:
        mapped_cat = pred_category

    return {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "predicted_reasoning": pred_reasoning,
        "predicted_info": pred_info,
        "predicted_category": pred_category,
        "mapped_predicted_info": pred_info,
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
    output_dir = "e:/nlp-for-disaster/exp2E-COT/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
