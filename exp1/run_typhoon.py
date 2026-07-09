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
    system_prompt = "You are an expert humanitarian disaster analyst with extensive experience in classifying disaster-related content. Your task is to classify tweets into a single specific category based on objective evidence rather than emotional responses."
    user_prompt = f"""Tweet: "{tweet_text}"

CLASSIFICATION CRITERIA:
Classify the tweet into exactly ONE of the following categories. Choose the most specific and dominant category represented in the text:

- not_informative: The tweet does NOT contain specific information, represents only emotions, prayers, general opinions, political comments, jokes, or is completely unrelated to disaster management.
- affected_individuals: Mentions displaced people, survivors, or evacuees who are affected but does NOT report deaths or injuries.
- infrastructure_and_utility_damage: References damaged buildings, roads, bridges, electricity, water lines, or other utilities.
- injured_or_dead_people: Reports specific numbers or accounts of injured, hospitalized, or deceased individuals.
- missing_or_found_people: Mentions people who are currently missing, search/rescue missions looking for individuals, or people who have been found/rescued.
- rescue_volunteering_or_donation_effort: Mentions relief goods, donation drives, financial aid, volunteer networks, or rescue team deployment.
- vehicle_damage: References damaged cars, trucks, buses, trains, or rescue vehicles.
- other_relevant_information: General informative reports such as weather forecasts, storm paths, satellite observations, or warnings without specific human or physical impact details.

Return classification by calling the specified function. You must call the function 'classify' with your prediction."""

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
                category = args.get("category")
                if category in CATEGORIES:
                    return category, prompt_tokens, completion_tokens, latency
                    
            text = choice.message.content or ""
            for cat in CATEGORIES:
                if cat in text.lower():
                    return cat, prompt_tokens, completion_tokens, latency
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
    
    pred_cat, in_tokens, out_tokens, latency = predict(tweet_text, temp)
    
    mapped_info = "not_informative" if pred_cat == "not_informative" else "informative"
    mapped_human = "not_humanitarian" if pred_cat == "not_informative" else pred_cat
    
    return {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        "true_text_info": true_info,
        "true_text_human": true_human,
        "predicted_category": pred_cat,
        "mapped_predicted_info": mapped_info,
        "mapped_predicted_category": mapped_human,
        "tweet_text_char_count": len(tweet_text),
        "token_in_use": in_tokens,
        "token_out_use": out_tokens,
        "latency_seconds": latency
    }

def main():
    df = pd.read_csv("e:/nlp-for-disaster/dataset/dataset_sample_500.csv")
    results = []
    
    # Check if 0.0 results already exist
    existing_0_csv = "e:/nlp-for-disaster/exp1/results/typhoon-v2.5_results.csv"
    temps_to_run = [0.1, 0.2, 0.3]
    if os.path.exists(existing_0_csv):
        print(f"[{model_name}] Found existing 0.0 results, loading them...")
        df_0 = pd.read_csv(existing_0_csv)
        df_0['temperature'] = 0.0
        cols_to_keep = ["tweet_id", "tweet_text", "true_text_info", "true_text_human", "predicted_category", 
                        "mapped_predicted_info", "mapped_predicted_category", "tweet_text_char_count", 
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
    output_dir = "e:/nlp-for-disaster/exp1/results"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"{model_name}_temp_results.csv"), index=False)
    print(f"Finished all temperatures for {model_name}.")

if __name__ == '__main__':
    main()
