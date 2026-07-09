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
    system_prompt = "You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.\n\nBIAS RULE: When in doubt, lean toward 'informative'. Only classify as 'not_informative' when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case."
    user_prompt = f"""Does this tweet contain SPECIFIC, FACTUAL information about a disaster event?

informative — Mark as informative if the tweet contains ANY of:
  - Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  - Named locations with concrete disaster impact described
  - Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  - Reports of rescue, aid, or emergency response operations with details
  - Measurable data: wind speed, magnitude, flood levels, temperature extremes

not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  - Pure emotional expression (prayers, sympathy, fear, grief) with NO factual details
  - Political argument or blame with NO specific disaster impact described
  - Jokes, obvious sarcasm, or clear misinformation
  - Completely off-topic content unrelated to the disaster
  - Vague awareness posts ("Thinking of everyone affected")

REFERENCE EXAMPLES (apply criteria above to any tweet, not just similar-looking ones):

[Clear not_informative] "Praying for everyone in the typhoon's path. God bless 🙏"
→ not_informative (pure emotional, no factual details)

[Clear informative] "6.2 Earthquake hits Nepal - 150 killed, rescue teams deployed"
→ informative (specific casualty count + response details)

[Edge case → informative] "PHOTOS: Deadly wildfires rage in California https://t.co/td9xT3vXOL"
→ informative (reports deadly wildfire with factual context, despite no exact number)

[Edge case → not_informative] "California wildfire. 4 https://t.co/a8oD5rkDdI"
→ not_informative (no specific factual content, just a fragment + link)

NOTE: Use the criteria above to classify any tweet regardless of similarity to these examples.

Tweet: "{tweet_text}"

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
                info = args.get("informativeness", "not_informative")
                return info, prompt_tokens, completion_tokens, latency
            return "not_informative", prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"Error Agent 1 {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "not_informative", 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def predict_agent2(tweet_text, temp, retries=5):
    system_prompt = "You are a disaster content categorizer. The tweet you receive has already been confirmed as containing specific disaster information. Your job is to identify its PRIMARY humanitarian content category.\n\nCORE PRINCIPLE: Choose the MOST SPECIFIC category that fits the tweet's dominant subject. Use 'other_relevant_information' only as a last resort when no other category applies."
    user_prompt = f"""This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category.

Apply these rules broadly across all disaster types — not just the examples below:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals even if displacement also mentioned

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout
   ⚠ Do NOT use when vehicles are the primary subject

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization
   Signal words: donate, volunteer, aid convoy, rescue team deployed, relief supplies, fundraising, emergency shelter opening
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, warnings without impact
   ⚠ If any specific category above fits, use that instead

REFERENCE EXAMPLES (multiple disaster types):

injured_or_dead_people → [Wildfire] "Mass Evacuations in California as Wildfires Kill at Least 10"
injured_or_dead_people → [Earthquake] "6.2 Earthquake hits Nepal - 150 killed, rescue teams deployed"
missing_or_found_people → [Wildfire] "More than 100 missing persons reports made in California wildfires"
missing_or_found_people → [Flood] "Family of 5 missing after flash flood swept through their home, search ongoing"
affected_individuals → [Wildfire] "I just had to evacuate my home in California due to the wildfire."
affected_individuals → [Hurricane] "Over 3000 evacuees sheltering at Houston Civic Center after Harvey flooding"
infrastructure_and_utility_damage → [Hurricane] "Power outage affecting 1.2 million homes in Florida after Hurricane Irma"
infrastructure_and_utility_damage → [Earthquake] "Multiple bridges collapsed in Kathmandu following 7.8 magnitude quake"
vehicle_damage → [Flood] "Dozens of vehicles swept away as flash flood overtook parking garage in Riyadh"
vehicle_damage → [Wildfire] "Cars burned with melted rims, trees standing — wildfire path"
rescue_volunteering_or_donation_effort → [Typhoon] "Red Cross deploying 500 relief workers to typhoon-hit provinces"
rescue_volunteering_or_donation_effort → [Wildfire] "How to help Napa fire victims: 8 things you can do for Wine Country right now"
other_relevant_information → [Hurricane] "Hurricane Maria now Category 4 with 130mph winds, expected to hit Puerto Rico Tuesday"
other_relevant_information → [Earthquake] "USGS: 7.1 magnitude earthquake detected off coast of Japan, tsunami warning issued"

NOTE: Apply the category definitions above to classify any tweet regardless of disaster type or similarity to examples.

Tweet: "{tweet_text}"

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
    tweet_text = row['tweet_text']
    true_info = row['text_info']
    true_human = row['text_human']
    
    pred_info, a1_in, a1_out, a1_lat = predict_agent1(tweet_text, temp)
    
    if pred_info == "informative":
        pred_cat, a2_in, a2_out, a2_lat = predict_agent2(tweet_text, temp)
        final_info = "informative"
        final_cat = pred_cat
    else:
        pred_cat = ""
        a2_in, a2_out, a2_lat = 0, 0, 0.0
        final_info = "not_informative"
        final_cat = "not_humanitarian"
        
    return {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "agent1_predicted_info": pred_info,
        "agent2_predicted_category": pred_cat,
        "final_predicted_info": final_info,
        "final_predicted_category": final_cat,
        "tweet_text_char_count": len(tweet_text),
        "token_in_use": a1_in + a2_in,
        "token_out_use": a1_out + a2_out,
        "agent1_latency_seconds": a1_lat,
        "agent2_latency_seconds": a2_lat,
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
    output_dir = "e:/nlp-for-disaster/exp3F/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
