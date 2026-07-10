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

agent1_tools = [
    {
        "type": "function",
        "function": {
            "name": "filter_informativeness",
            "description": "Decide whether a tweet contains specific disaster information with reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_reasoning": {
                        "type": "string",
                        "description": "Briefly analyze the raw tweet for disaster indicators (e.g. casualty counts, location impacts, physical damage) and explain your informativeness logic."
                    },
                    "informativeness": {
                        "type": "string",
                        "enum": ["informative", "not_informative"],
                        "description": "Is the tweet informative about a disaster?"
                    }
                },
                "required": ["short_reasoning", "informativeness"]
            }
        }
    }
]

agent2_tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_category",
            "description": "Categorize the disaster tweet into a humanitarian class with reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_reasoning": {
                        "type": "string",
                        "description": "Provide a brief 1-2 sentence analysis connecting key tweet clues to the selected humanitarian category."
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "injured_or_dead_people",
                            "missing_or_found_people",
                            "affected_individuals",
                            "infrastructure_and_utility_damage",
                            "vehicle_damage",
                            "rescue_volunteering_or_donation_effort",
                            "other_relevant_information"
                        ],
                        "description": "The primary dominant humanitarian category representing the tweet."
                    }
                },
                "required": ["short_reasoning", "category"]
            }
        }
    }
]

def predict_agent1(tweet_text, temp, retries=5):
    system_prompt = "You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.\n\nBIAS RULE: When in doubt, lean toward 'informative'. Only classify as 'not_informative' when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case."
    user_prompt = f"""Tweet: "{tweet_text}"

Does this tweet contain SPECIFIC, FACTUAL information about a disaster event? Call 'filter_informativeness' with your reasoning and choice.

informative — Mark as informative if the tweet contains ANY of:
  - Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  - Named locations with concrete disaster impact described
  - Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  - Reports of rescue, aid, or emergency response operations with details
  - Measurable data: wind speed, magnitude, flood levels, temperature extremes
  - Weather forecasts, warnings, storm tracks, magnitude reports, or direct discussion referencing a specific disaster (e.g., "Prayers for Nepal #earthquake" contains the Nepal earthquake keyword).

not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  - Pure emotional expression (prayers, sympathy, fear, grief) with NO specific disaster references or details (e.g., "Thinking of everyone affected, stay safe").
  - Political argument or blame with NO specific disaster impact described.
  - Jokes, obvious sarcasm, or clear misinformation.
  - Completely off-topic content unrelated to the disaster.

STEPS:
1. Under "short_reasoning", identify critical words and explain why the tweet is informative or not in 1-2 sentences.
2. Call the function 'filter_informativeness' with both your reasoning and informativeness decision."""

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
                reasoning = args.get("short_reasoning", "")
                info = args.get("informativeness", "not_informative")
                return reasoning, info, prompt_tokens, completion_tokens, latency

            return "", "not_informative", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Agent 1 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "", "not_informative", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def predict_agent2(tweet_text, temp, retries=5):
    system_prompt = "You are a disaster content categorizer. The tweet you receive has already been confirmed as containing specific disaster information. Your job is to explain your analysis and identify its PRIMARY humanitarian content category.\n\nCORE PRINCIPLE: Choose the MOST SPECIFIC category that fits the tweet's dominant subject. Use 'other_relevant_information' only as a last resort when no other category applies."
    user_prompt = f"""Tweet: "{tweet_text}"

This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category using this priority order:

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

EDGE-CASE RESOLUTION RULES:
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

STEPS:
1. Under "short_reasoning", explain your reasoning connecting the tweet text clues to one of the categories in 1-2 sentences.
2. Call 'classify_category' with both your reasoning and chosen category."""

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
                reasoning = args.get("short_reasoning", "")
                category = args.get("category", "other_relevant_information")
                return reasoning, category, prompt_tokens, completion_tokens, latency

            return "", "other_relevant_information", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Agent 2 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "", "other_relevant_information", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def process_row(row, temp):
    tweet_id = row['tweet_id']
    tweet_text = row['tweet_text']
    true_info = row['text_info']
    true_human = row['text_human']

    # Agent 1 Informativeness Filter
    pred_info_reason, pred_info, a1_in, a1_out, a1_lat = predict_agent1(tweet_text, temp)

    if pred_info == "informative":
        # Agent 2 Category Classifier
        pred_cat_reason, pred_cat, a2_in, a2_out, a2_lat = predict_agent2(tweet_text, temp)
        final_info = "informative"
        final_cat = pred_cat
    else:
        pred_cat_reason = ""
        pred_cat = ""
        a2_in, a2_out, a2_lat = 0, 0, 0.0
        final_info = "not_informative"
        final_cat = "not_humanitarian"

    return {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "predicted_info_reasoning": pred_info_reason,
        "predicted_info": pred_info,
        "predicted_cat_reasoning": pred_cat_reason,
        "predicted_category": pred_cat,
        "mapped_predicted_info": final_info,
        "mapped_predicted_category": final_cat,
        "tweet_text_char_count": len(tweet_text),
        "token_in_use": a1_in + a2_in,
        "token_out_use": a1_out + a2_out,
        "latency_seconds": a1_lat + a2_lat
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
    output_dir = "e:/nlp-for-disaster/exp3E-COT/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
