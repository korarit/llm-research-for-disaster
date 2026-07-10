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
            "description": "Classify the disaster-related tweet and provide a brief reasoning analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_reasoning": {
                        "type": "string",
                        "description": "Briefly analyze the tweet text to identify critical clues (e.g. casualties, evacuation, collapsed) and explain in 1-2 sentences why it belongs to the chosen category."
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "The primary dominant humanitarian category representing the tweet."
                    }
                },
                "required": ["short_reasoning", "category"]
            }
        }
    }
]

def predict(tweet_text, temp, retries=5):
    system_prompt = """You are a humanitarian disaster information analyst. Your task is to analyze social media posts (tweets) collected during disasters, explain your reasoning, and classify them into exactly one category for emergency response."""

    user_prompt = f"""Tweet: "{tweet_text}"

Classify this tweet into the SINGLE most dominant humanitarian category using this priority order:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals and infrastructure damage if any death or injury is mentioned.

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment.

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned.

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout, structural damage
   ⚠ Do NOT use when vehicles are the primary subject.

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention.

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization, emergency helpline sharing, relief distribution
   Signal words: donate, volunteers, aid, rescue team, relief, relief goods, aid distribution
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons.

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, official warnings without impact details, general news/opinions/expressions of solidarity that mention a specific disaster.
   ⚠ If any specific category above fits, use that instead.

8. not_informative
   Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.

EDGE-CASE RESOLUTION RULES:
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

STEPS:
1. Under "short_reasoning", identify critical words (e.g. death counts, damaged roads) and explain why the tweet belongs to your chosen category in 1-2 sentences.
2. Call the function 'classify' with both your reasoning and chosen category."""

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
                reasoning = args.get("short_reasoning", "")
                category = args.get("category", "not_informative")
                return reasoning, category, prompt_tokens, completion_tokens, latency

            return "", "not_informative", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "", "not_informative", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    tweet_text = row['tweet_text']
    true_info = row['text_info']
    true_human = row['text_human']

    pred_reasoning, pred_category, in_tokens, out_tokens, latency = predict(tweet_text, temp)

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
        "predicted_reasoning": pred_reasoning,
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
    output_dir = "e:/nlp-for-disaster/exp1E-COT/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
