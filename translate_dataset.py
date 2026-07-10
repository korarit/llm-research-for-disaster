import os
import time
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Config
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if not OPEN_ROUTER_KEY:
    raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables!")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_KEY
)

model_id = "google/gemini-3.1-flash-lite"
input_file = "e:/nlp-for-disaster/dataset/dataset_sample_1000.csv"
output_file = "e:/nlp-for-disaster/dataset/dataset_sample_1000_traslate_th.csv"

SYSTEM_PROMPT = """You are an expert native Thai translator and disaster response analyst. Your task is to translate English disaster-related tweets into natural, fluent, and organic Thai as spoken by real Thai social media users (on platforms like Twitter/X and Facebook) during emergencies.

CRITICAL RULES FOR TRANSLATION:
1. DO NOT translate word-for-word (literal translation). Avoid English grammar structures in Thai (e.g., avoid excessive passive voice like "ถูกทำลายโดย..." unless natural).
2. Keep the translation conversational, casual, and natural. AVOID formal, academic, or overly journalistic/news-anchor Thai.
   - Do NOT use words like "เผย" (revealed), "ระบุว่า" (stated that), "ดำเนินงาน" (proceed), "ทำการ" (perform), "ส่งผลให้" (result in), or "ยังคง" (still) unless it fits a natural casual style.
   - Use natural connective and reporting verbs: e.g., use "บอกว่า" or "แจ้งว่า" instead of "เผย/ระบุ". Use "เพราะ" or "เนื่องจาก" instead of "ส่งผลให้".
   - DO NOT automatically append polite particles like "ครับ" (krub) or "ค่ะ" (kha) at the end of sentences. Most real tweets do not use them. Match the tone and level of formality of the original tweet.
   - AVOID overly conversational or playful particles like "นะ", "จ้า", "เนี่ย", "ดิ" or "เด้อ" unless the original tweet is clearly written in a playful/intimate personal chat style. Disaster-related social media posts (even casual ones) are informative and serious, so adding "นะ" at the end of reports sounds out of place and unnatural.
3. DO NOT change, add, or omit any crucial facts, numbers, datetime, or details (except locations, which must be localized to Thai places).
   - For example, if the English tweet mentions "10 injured", the Thai translation must report "บาดเจ็บ 10 คน" or "เจ็บ 10 คน" (avoid overly formal "บาดเจ็บ 10 ราย" if it sounds like an official police report).
   - DO NOT assume or invent details not present in the original text. For example, if the tweet says children are "missing", translate it as "สูญหาย" or "หายตัวไป". Do NOT assume or translate that they are "ติดอยู่ใต้ซาก" (trapped under rubble) unless the English text explicitly says "trapped under rubble/debris".
4. Preserve the context of the classification classes:
   - If the original mentions physical damage (houses, roads, bridges), ensure the Thai translation makes it very clear and descriptive of infrastructure damage (e.g., "ถนนขาด", "เสาไฟล้ม", "บ้านพัง").
   - If the original is a rescue/donation request, make sure the Thai translation sounds like a natural call for help or coordinate donation.
5. Localize English names of locations to appropriate Thai places (provinces, districts, streets) instead of transliterating them, to make the text sound completely organic to Thailand (e.g., change "Houston" or "California" to places like "เชียงราย", "อุบลราชธานี", "พะเยา", "สาย 304" depending on what fits the disaster type naturally).

Guidelines & Examples:

Example 1:
- English: "Please pray for Houston. My house is flooded and we need immediate rescue."
- Bad Literal / Too Formal: "โปรดสวดอ้อนวอนเพื่อฮิวสตัน บ้านของฉันถูกน้ำท่วมและเราต้องการการกู้ภัยทันที (แปลทื่อ) / ขอแรงใจให้เชียงรายด้วยครับ ตอนนี้บ้านผมน้ำท่วมสูงมาก ต้องการความช่วยเหลือด่วนครับ (ทางการ/สุภาพเกินไป)"
- Good Natural & Localized: "ช่วยส่งใจ/ภาวนาให้เชียงรายด้วยนะ ตอนนี้บ้านน้ำท่วมสูงมาก อยากได้กู้ภัยเข้ามาช่วยด่วนเลย"

Example 2:
- English: "Red Cross volunteers are distributing food packs to 200 hurricane victims in Florida."
- Bad Literal / Too Formal: "อาสาสมัครสภากาชาดกำลังแจกจ่ายแพ็กเกจอาหารให้กับเหยื่อพายุเฮอริเคน 200 รายในฟลอริดา"
- Good Natural & Localized: "อาสากาชาดกำลังเอาอาหารและถุงยังชีพไปแจกช่วยเหลือผู้ประสบภัยพายุ 200 คนแถวสุราษฎร์"

Example 3:
- English: "Bridge on Route 9 collapsed due to the flash flood. Road closed."
- Bad Literal / Too Formal: "สะพานบนเส้นทาง 9 ทรุดตัวเนื่องจากน้ำท่วมฉับพลัน ถนนปิด (แปลทื่อ) / สะพานตรงถนนสาย 9 ขาดพังถล่มจากน้ำป่าไหลหลาก ตอนนี้ปิดการจราจรแล้วครับ (ทางการเกินไป)"
- Good Natural & Localized: "สะพานตรงถนนสาย 9 ขาดเพราะน้ำป่าไหลหลาก ตอนนี้ปิดถนนไปแล้ว"

Example 4:
- English: "Relatives say children missing after a school collapsed in Mexico's deadly earthquake have sent WhatsApp messages."
- Bad / Too Playful / Inaccurate: "ญาติเผย เด็กๆ ที่ติดอยู่ใต้ซากอาคารเรียนหลังถล่มจากเหตุแผ่นดินไหวรุนแรง ยังคงส่งข้อความผ่าน WhatsApp ออกมาได้ครับ (ทางการ/สำนวนข่าวหนังสือพิมพ์เกินไป) / ญาติๆ บอกว่าเด็กที่ยังติดอยู่ใต้ซากโรงเรียนถล่มจากเหตุแผ่นดินไหว ส่งข้อความทาง WhatsApp ออกมาได้แล้วนะ (แปลเกินจริงเรื่องติดใต้ซาก และใช้คำลงท้าย 'นะ' ที่ผิดกาลเทศะของการแจ้งข่าวภัยพิบัติ)"
- Good Natural & Localized: "ญาติบอกว่าเด็กๆ ที่สูญหายหลังโรงเรียนถล่มเพราะแผ่นดินไหวรุนแรงที่เม็กซิโก ส่งข้อความผ่าน WhatsApp มาแล้ว"
"""

def translate_tweet(tweet_text, retries=5):
    user_prompt = f"""Translate the following English tweet into natural Thai according to the translation guidelines.

Tweet to translate:
"{tweet_text}"

Return only the translated Thai text. Do not add any introduction or explanation."""

    backoff = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=256
            )
            translation = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting or quotes
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1].strip()
            if translation.startswith("'") and translation.endswith("'"):
                translation = translation[1:-1].strip()
            return translation
        except Exception as e:
            print(f"Error translating (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return f"[Translation Error: {e}]"
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0

def main():
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    # Load completed translations if file exists
    completed_translations = {}
    if os.path.exists(output_file):
        try:
            completed_df = pd.read_csv(output_file)
            completed_translations = dict(zip(completed_df['tweet_id'], completed_df['translated_thai']))
            print(f"Loaded existing output file. Found {len(completed_translations)} already translated rows.")
        except Exception as e:
            print(f"Could not load existing output file: {e}. Starting fresh.")
            
    # Filter rows to translate
    to_translate = []
    for idx, row in df.iterrows():
        tweet_id = row['tweet_id']
        if tweet_id in completed_translations:
            continue
        to_translate.append(row)
        
    print(f"Rows remaining to translate: {len(to_translate)}")
    if len(to_translate) == 0:
        print("All rows are already translated!")
        return

    # Translate using ThreadPoolExecutor for speed
    results = []
    max_workers = 15
    print(f"Starting translation using {max_workers} worker threads...")
    
    counter = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(translate_tweet, row['tweet_text']): row for row in to_translate
        }
        
        for future in as_completed(futures):
            row = futures[future]
            tweet_id = row['tweet_id']
            tweet_text = row['tweet_text']
            
            try:
                translated_text = future.result()
            except Exception as e:
                translated_text = f"[Exception Error: {e}]"
                
            completed_translations[tweet_id] = translated_text
            counter += 1
            
            if counter % 50 == 0 or counter == len(to_translate):
                print(f"Progress: {counter}/{len(to_translate)} tweets translated.")
                # Periodically save progress to avoid data loss
                save_df = df.copy()
                save_df['original_english'] = save_df['tweet_text']
                save_df['translated_thai'] = save_df['tweet_id'].map(completed_translations)
                
                # Rename columns according to the plan schema:
                # tweet_id, original_english, translated_thai, true_text_info, true_text_human, image_human, disaster_source
                save_df = save_df.rename(columns={
                    'text_info': 'true_text_info',
                    'text_human': 'true_text_human'
                })
                
                cols_to_save = ['tweet_id', 'original_english', 'translated_thai', 'true_text_info', 'true_text_human', 'image_human', 'disaster_source']
                save_df = save_df[cols_to_save]
                save_df.to_csv(output_file, index=False)
                print(f"Progress saved to {output_file}")

    print("\nTranslation complete!")
    print(f"Final file verified and saved to {output_file}")

if __name__ == '__main__':
    main()
