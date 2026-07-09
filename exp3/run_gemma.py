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
    system_prompt = "You are an expert humanitarian disaster analyst. Your task is to analyze tweets and determine if they contain specific information about disaster impact or response efforts."
    user_prompt = f"""Tweet: "{tweet_text}"

CLASSIFICATION CRITERIA:
Determine if the tweet contains SPECIFIC information:
- informative: Contains SPECIFIC disaster impact/response evidence, facts, or details (such as reports of damage, injuries, rescue activities, weather updates, donation needs).
- not_informative: Generic statements, emotions only (prayers, condolences), political arguments, jokes, or completely unrelated content.

Return classification by calling the specified function. You must call the function 'filter_informativeness' with your prediction."""

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
    system_prompt = "You are an expert humanitarian disaster analyst. Your task is to classify a disaster-related tweet into the dominant humanitarian category based on objective evidence."
    user_prompt = f"""Tweet: "{tweet_text}"

CLASSIFICATION CRITERIA:
Identify the DOMINANT content of this disaster-related tweet. Choose exactly ONE category:
- affected_individuals: Mentions displaced people, survivors, emotional responses (NOT injured/dead).
- infrastructure_and_utility_damage: References damaged buildings, roads, bridges, power/water utilities.
- injured_or_dead_people: Reports injuries, deaths, or specific casualty numbers.
- missing_or_found_people: Mentions people who are missing, found, or rescued by name or count.
- other_relevant_information: Weather data, satellite images, warning alerts without specific physical/human impact.
- rescue_volunteering_or_donation_effort: Mentions donations, rescue missions, aid, volunteers.
- vehicle_damage: References damaged cars, trucks, ambulances, buses.

Return classification by calling the specified function. You must call the function 'classify_category' with your prediction."""

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
    
    existing_0_csv = "e:/nlp-for-disaster/exp3/results/gemma-4_results.csv"
    temps_to_run = [0.1, 0.2, 0.3]
    if os.path.exists(existing_0_csv):
        print(f"[{model_name}] Found existing 0.0 results, loading them...")
        df_0 = pd.read_csv(existing_0_csv)
        df_0['temperature'] = 0.0
        cols_to_keep = ["tweet_id", "tweet_text", "true_text_info", "true_text_human", "agent1_predicted_info", 
                        "agent2_predicted_category", "final_predicted_info", "final_predicted_category", 
                        "tweet_text_char_count", "token_in_use", "token_out_use", "agent1_latency_seconds", 
                        "agent2_latency_seconds", "latency_seconds", "temperature"]
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
    output_dir = "e:/nlp-for-disaster/exp3/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
