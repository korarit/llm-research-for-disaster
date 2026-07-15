import os
import time
import json
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

PREFIXES_FEMALE = ["นาง", "นางสาว", "คุณ", "พี่", "น้อง", "ป้า", "ยาย", "เจ๊", "น้า", "อา", "หมอ"]
PREFIXES_MALE = ["นาย", "คุณ", "พี่", "น้อง", "ลุง", "ตา", "เฮีย", "น้า", "อา", "หมอ"]

def load_names_data():
    """Load first names (male and female), nicknames, and family names from dataset/thai_name/ files."""
    female_names = []
    male_names = []
    last_names = []
    female_nicknames = []
    male_nicknames = []
    
    # Try relative path first, then absolute
    thai_name_dir = "dataset/thai_name"
    if not os.path.exists(thai_name_dir):
        thai_name_dir = "e:/nlp-for-disaster/dataset/thai_name"
        
    female_path = os.path.join(thai_name_dir, "female_names_th.txt")
    male_path = os.path.join(thai_name_dir, "male_names_th.txt")
    family_path = os.path.join(thai_name_dir, "family_names_th.txt")
    female_nick_path = os.path.join(thai_name_dir, "nickname_female_th.txt")
    male_nick_path = os.path.join(thai_name_dir, "nickname_male_th.txt")
    
    try:
        if os.path.exists(female_path):
            with open(female_path, "r", encoding="utf-8") as f:
                female_names.extend([line.strip() for line in f if line.strip()])
        if os.path.exists(male_path):
            with open(male_path, "r", encoding="utf-8") as f:
                male_names.extend([line.strip() for line in f if line.strip()])
        if os.path.exists(family_path):
            with open(family_path, "r", encoding="utf-8") as f:
                last_names.extend([line.strip() for line in f if line.strip()])
        if os.path.exists(female_nick_path):
            with open(female_nick_path, "r", encoding="utf-8") as f:
                female_nicknames.extend([line.strip() for line in f if line.strip()])
        if os.path.exists(male_nick_path):
            with open(male_nick_path, "r", encoding="utf-8") as f:
                male_nicknames.extend([line.strip() for line in f if line.strip()])
    except Exception as e:
        print(f"Error loading name files: {e}")
        
    female_names = list(set([n for n in female_names if n]))
    male_names = list(set([n for n in male_names if n]))
    last_names = list(set([n for n in last_names if n]))
    female_nicknames = list(set([n for n in female_nicknames if n]))
    male_nicknames = list(set([n for n in male_nicknames if n]))
    
    # Fallbacks if files are missing or empty
    if not female_names:
        female_names = ["สมศรี", "ป้าดา", "ยายแม้น", "เจ๊พร", "แดง", "กานต์", "สมเกียรติ"]
    if not male_names:
        male_names = ["สมชาย", "ลุงป้อม", "สมศักดิ์", "วิชิต", "แดง", "กานต์", "สมเกียรติ"]
    if not last_names:
        last_names = ["ใจดี", "กองแก้ว", "ทองดี", "รักชาติ", "มั่นคง"]
    if not female_nicknames:
        female_nicknames = ["กิ๊ฟ", "ฝ้าย", "ส้ม", "แป้ง", "ก้อย"]
    if not male_nicknames:
        male_nicknames = ["เบียร์", "บอม", "แบงค์", "แม็ค", "เก่ง"]
        
    return female_names, male_names, last_names, female_nicknames, male_nicknames

# Load name lists at startup
FEMALE_NAMES, MALE_NAMES, LAST_NAMES, FEMALE_NICKNAMES, MALE_NICKNAMES = load_names_data()

def load_symptoms_data():
    """Load symptom banks from dataset/sample_iitt_thai folder."""
    symptoms = {
        "child": {
            "RED": [],
            "YELLOW": [],
            "GREEN": []
        },
        "adult": {
            "RED": [],
            "YELLOW": [],
            "GREEN": []
        }
    }
    
    # Try relative path first, then absolute
    base_dir = "dataset/sample_iitt_thai"
    if not os.path.exists(base_dir):
        base_dir = "e:/nlp-for-disaster/dataset/sample_iitt_thai"
        
    for age_group in ["child", "adult"]:
        for triage_color in ["RED", "YELLOW", "GREEN"]:
            filename = f"{age_group}_{triage_color.lower()}.csv"
            filepath = os.path.join(base_dir, filename)
            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    if "symptom" in df.columns:
                        symptoms[age_group][triage_color] = [s.strip() for s in df["symptom"].tolist() if s.strip()]
            except Exception as e:
                print(f"Error loading symptom file {filename}: {e}")
                
            # Fallbacks in case file is missing or empty
            if not symptoms[age_group][triage_color]:
                print(f"Warning: Using fallback symptoms for {age_group} {triage_color}")
                if triage_color == "RED":
                    symptoms[age_group][triage_color] = ["หมดสติ ปลุกไม่ตื่น", "หายใจเหนื่อยหอบตัวเขียว", "ชักเกร็ง ตาเหลือก"]
                elif triage_color == "YELLOW":
                    symptoms[age_group][triage_color] = ["ขาหัก ผิดรูป", "อาเจียนตลอดเวลา", "มีไข้สูง ซึม"]
                else:
                    symptoms[age_group][triage_color] = ["แผลถลอกเล็กน้อย", "มีไข้ต่ำๆ เดินได้", "ไอนิดหน่อย"]
                    
    return symptoms

SYMPTOMS_BANK = load_symptoms_data()

def generate_random_name(gender=None):
    """Generate a Thai name info dictionary containing prefix, first name, last name, nickname, full name, and gender."""
    if not gender:
        gender = random.choice(["female", "male"])
        
    if gender == "female":
        first_name = random.choice(FEMALE_NAMES)
        nickname = random.choice(FEMALE_NICKNAMES) if random.random() < 0.5 else None
        prefix = random.choice(PREFIXES_FEMALE) if random.random() < 0.5 else None
    else:
        first_name = random.choice(MALE_NAMES)
        nickname = random.choice(MALE_NICKNAMES) if random.random() < 0.5 else None
        prefix = random.choice(PREFIXES_MALE) if random.random() < 0.5 else None
        
    last_name = random.choice(LAST_NAMES) if random.random() < 0.5 else None
    
    # Sometimes just refer to them by their nickname in social media text (20% chance)
    use_only_nickname = random.random() < 0.2 and nickname is not None
    if use_only_nickname:
        full_name = nickname
        if prefix:
            full_name = prefix + nickname
        first_name = nickname
        last_name = None
    else:
        full_name = ""
        if prefix:
            full_name += prefix
        full_name += first_name
        if last_name:
            full_name += " " + last_name
            
    return {
        "name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "nickname": nickname,
        "prefix": prefix,
        "gender": gender,
        "use_only_nickname": use_only_nickname
    }


# Initialize clients
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

if not TYPHOON_API_KEY or not OPEN_ROUTER_API_KEY:
    raise ValueError("Missing TYPHOON_API_KEY or OPEN_ROUTER_API_KEY in .env file!")

typhoon_client = OpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=TYPHOON_API_KEY
)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY
)

def generate_thai_phone():
    """Generate a random Thai mobile phone number in raw, dash, or space formatting."""
    prefixes = ["08", "09", "06"]
    prefix = random.choice(prefixes)
    rest = "".join([str(random.randint(0, 9)) for _ in range(8)])
    phone = prefix + rest
    
    fmt = random.choice(["raw", "dash", "space"])
    if fmt == "dash":
        return f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
    elif fmt == "space":
        return f"{phone[:3]} {phone[3:6]} {phone[6:]}"
    else:
        return phone

def clean_phone(phone_str):
    """Normalize phone number by removing spaces, hyphens, and converting country code."""
    if not phone_str:
        return ""
    cleaned = "".join(c for c in phone_str if c.isdigit())
    if cleaned.startswith("66"):
        cleaned = "0" + cleaned[2:]
    return cleaned

def generate_location_pool(client, num_needed=350):
    """Pre-generate a pool of unique Thai location names using Typhoon to avoid rate limits and save time."""
    system_instruction = """You are an expert in Thai geography and local landmarks. Your task is to generate a list of realistic, natural-sounding location names in Thailand for a disaster situation (such as a flooded street, a local temple, a village name, or a sub-district).

Format instructions:
- Focus on provinces prone to floods (e.g., Chiang Rai, Chiang Mai, Phrae, Nan, Sukhothai, Ubon Ratchathani).
- Do NOT generate locations in Hat Yai.
- Return ONLY a JSON list of strings containing the generated location names in Thai (each including street, village, and/or landmark with sub-district/province), without any explanations.
- Output valid JSON format only, like: ["สถานที่ 1", "สถานที่ 2", ...]"""

    # Fallback pool in case Typhoon fails or returns invalid format
    fallback_pool = [
        "ซอย 4 เหมืองแดง แม่สาย เชียงราย",
        "บ้านห้วยทราย ต.แม่ยาว อ.เมืองเชียงราย",
        "ชุมชนท่ากอไผ่ อ.วารินชำราบ อุบลราชธานี",
        "ซอยเหมืองแดง 3 ต.แม่สาย อ.แม่สาย เชียงราย",
        "บ้านแม่ต๋ำ ต.ท่าก๊อ อ.แม่สรวย เชียงราย",
        "ต.เวียง อ.เมืองเชียงราย เชียงราย",
        "ต.เวียงพางคำ อ.แม่สาย เชียงราย",
        "บ้านหนองบัว ต.แม่เจดีย์ อ.เวียงป่าเป้า เชียงราย",
        "ชุมชนสองแคว ต.เวียง อ.เมืองเชียงใหม่ เชียงใหม่",
        "บ้านริมน้ำ ต.ป่าแดด อ.เมืองเชียงใหม่ เชียงใหม่",
        "บ้านหนองตอง ต.หนองตอง อ.หางดง เชียงใหม่",
        "ต.ช้างคลาน อ.เมืองเชียงใหม่ เชียงใหม่",
        "ถนนเจริญประเทศ อ.เมืองเชียงใหม่ เชียงใหม่",
        "บ้านสบกอน ต.เชียงกลาง อ.เชียงกลาง น่าน",
        "บ้านดอนศรีเสริม ต.ในเวียง อ.เมืองน่าน น่าน",
        "ต.ฝายแก้ว อ.ภูเพียง น่าน",
        "ต.ดอนแก้ว อ.สบปราบ ลำปาง",
        "บ้านป่ารวก ต.แม่ยม อ.เมืองแพร่ แพร่",
        "ต.ในเวียง อ.เมืองแพร่ แพร่",
        "บ้านวังธง ต.วังธง อ.เมืองแพร่ แพร่",
        "ต.ป่าแมต อ.เมืองแพร่ แพร่",
        "ต.ปากยม อ.สวรรคโลก สุโขทัย",
        "บ้านท่าทอง ต.เมืองบางยม อ.สวรรคโลก สุโขทัย",
        "ต.ธานี อ.เมืองสุโขทัย สุโขทัย",
        "ต.ปากแคว อ.เมืองสุโขทัย สุโขทัย",
        "บ้านไร่ ต.สามเรือน อ.ศรีสำโรง สุโขทัย",
        "ชุมชนหลังวัดป่าอุบลแก้ว ต.ในเมือง อ.เมืองอุบลราชธานี อุบลราชธานี",
        "บ้านท่าบ้งมั่ง ต.วารินชำราบ อ.วารินชำราบ อุบลราชธานี",
        "ต.ในเมือง อ.เมืองอุบลราชธานี อุบลราชธานี",
        "บ้านกุดระงุม ต.บุ่งไหม อ.วารินชำราบ อุบลราชธานี",
        "ชุมชนเกดแก้ว ต.วารินชำราบ อ.วารินชำราบ อุบลราชธานี"
    ]
    
    import re
    
    def repair_json_list(text):
        """Extract and repair any potentially truncated or malformed JSON list of strings."""
        text = text.strip()
        
        # Try direct parsing first
        try:
            val = json.loads(text)
            if isinstance(val, list):
                return val
        except Exception:
            pass
            
        # Regex to find all valid quoted strings
        # Matches double quoted strings correctly handling escapes
        matches = list(re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text))
        if matches:
            return [m.group(1) for m in matches]
            
        return []
        
    pool = []
    chunk_size = 30
    chunks = (num_needed + chunk_size - 1) // chunk_size
    
    for i in range(chunks):
        print(f"Generating location pool chunk {i+1}/{chunks}...")
        backoff = 1.0
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="typhoon-v2.5-30b-a3b-instruct",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Generate a JSON list of exactly {chunk_size} unique Thai location names for flood disaster reports."}
                    ],
                    temperature=1.0,
                    max_tokens=2048
                )
                content = response.choices[0].message.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                data = repair_json_list(content)
                if data:
                    pool.extend(data)
                    break
                else:
                    raise ValueError("No valid location strings found in LLM response")
            except Exception as e:
                print(f"Error generating location pool chunk (attempt {attempt+1}): {e}")
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2.0
                
    pool = [p.strip() for p in pool if isinstance(p, str) and p.strip()]
    if not pool:
        print("Using fallback location pool...")
        pool = fallback_pool
    else:
        pool = list(set(pool))
        if len(pool) < num_needed:
            pool.extend(fallback_pool)
            
    random.shuffle(pool)
    print(f"Generated a pool of {len(pool)} unique locations.")
    return pool

def generate_random_parameters(location_pool):
    """Generate ground truth parameters randomly based on plan_dataset_ner.md configurations."""
    # 1. Contacts
    # Decide if reporter is the same person as the victim (40% chance)
    reporter_is_victim = random.random() < 0.4
    
    victim_name_info = generate_random_name()
    if reporter_is_victim:
        reporter_name_info = victim_name_info
        victim_phone = generate_thai_phone()
        reporter_phone = victim_phone
        # Either both have the phone (80% chance) or neither does (20% chance)
        if random.random() < 0.2:
            victim_phone = None
            reporter_phone = None
    else:
        reporter_name_info = generate_random_name()
        while reporter_name_info["first_name"] == victim_name_info["first_name"]:
            reporter_name_info = generate_random_name()
            
        victim_phone = generate_thai_phone()
        reporter_phone = generate_thai_phone()
        
        # Phone scenarios
        scenario = random.choice(["A", "B", "C", "D"])
        if scenario == "B":
            reporter_phone = None
        elif scenario == "C":
            victim_phone = None
        elif scenario == "D":
            victim_phone = None
            reporter_phone = None
        
    # Generate victims list (1 to 3 victims)
    num_victims = random.randint(1, 3)
    victims_list = []
    
    for idx in range(num_victims):
        # Age group: 35% child, 65% adult
        age_group = "child" if random.random() < 0.35 else "adult"
        
        # Age
        if age_group == "child":
            age = random.randint(1, 11)
        else:
            # 50% chance of specifying age, 50% null
            age = random.randint(12, 85) if random.random() < 0.5 else None
            
        # Age disclosure: direct (age number specified) vs indirect (described using age-group words)
        if age is not None:
            age_disclosure = "direct" if random.random() < 0.5 else "indirect"
        else:
            age_disclosure = "indirect"
            
        # Triage color
        triage_color = random.choice(["RED", "YELLOW", "GREEN"])
        
        # Name: 30% chance of a specific name, 70% null
        gender = random.choice(["female", "male"])
        name = None
        if random.random() < 0.3:
            name_info = generate_random_name(gender=gender)
            if age_group == "child":
                name = f"น้อง{name_info['first_name']}"
            else:
                prefix_pool = ["คุณ", "พี่", "น้า", "ป้า", "ยาย", "เจ๊"] if gender == "female" else ["คุณ", "พี่", "น้า", "ลุง", "ตา", "เฮีย"]
                prefix = random.choice(prefix_pool)
                name = f"{prefix}{name_info['first_name']}"
        
        # Special case: map one victim's name to the contact_victim's name if we have a name and want to link it
        if idx == 0 and name is not None:
            name = victim_name_info["name"]
            gender = victim_name_info["gender"]
            
        # Get symptom from loaded bank
        symptoms_pool = SYMPTOMS_BANK[age_group][triage_color]
        symptoms_literal = random.choice(symptoms_pool) if symptoms_pool else ""
        
        victims_list.append({
            "name": name,
            "age": age,
            "age_group": age_group,
            "age_disclosure": age_disclosure,
            "gender": gender,
            "triage_color": triage_color,
            "symptoms_literal": symptoms_literal
        })
        
    # Deriving aggregate counts
    dead = 1 if random.random() < 0.08 else 0  # 8% chance of 1 dead victim
    critical = sum(1 for v in victims_list if v["triage_color"] == "RED")
    urgent = sum(1 for v in victims_list if v["triage_color"] == "YELLOW")
    safe = sum(1 for v in victims_list if v["triage_color"] == "GREEN")
    child = sum(1 for v in victims_list if v["age_group"] == "child")
    infant = sum(1 for v in victims_list if v["age"] is not None and v["age"] <= 1)
    
    # 3. Items Needed (50% chance of 0, 30% chance of 1, 20% chance of 2-20)
    def get_item():
        r = random.random()
        if r < 0.5:
            return 0
        elif r < 0.8:
            return 1
        return random.randint(2, 20)
        
    firstAid = get_item()
    food = get_item()
    energy = get_item()
    
    # 4. Coordinates & Location (7 Combinations)
    loc_scenario = random.randint(1, 7)
    has_location_name = loc_scenario in [1, 4, 5, 7]
    has_google_map = loc_scenario in [2, 4, 6, 7]
    has_coordinates = loc_scenario in [3, 5, 6, 7]
    
    location_name = None
    if has_location_name:
        location_name = location_pool.pop() if location_pool else "ซอย 4 เหมืองแดง แม่สาย เชียงราย"
        
    google_map_url = None
    if has_google_map:
        random_suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10))
        google_map_url = f"https://maps.app.goo.gl/{random_suffix}"
        
    lat = 0.0
    lng = 0.0
    if has_coordinates:
        lat = round(random.uniform(13.0, 20.4), 4)  # Upper/central flood regions
        lng = round(random.uniform(98.5, 105.0), 4)
        
    return {
        "location": {
            "location_name": location_name,
            "google_map_url": google_map_url,
            "lat": lat,
            "lng": lng
        },
        "contact_victim": {
            "name": victim_name_info["name"],
            "first_name": victim_name_info["first_name"],
            "last_name": victim_name_info["last_name"],
            "prefix": victim_name_info["prefix"],
            "nickname": victim_name_info["nickname"],
            "gender": victim_name_info["gender"],
            "phone": victim_phone
        },
        "contact_reporter": {
            "name": reporter_name_info["name"],
            "first_name": reporter_name_info["first_name"],
            "last_name": reporter_name_info["last_name"],
            "prefix": reporter_name_info["prefix"],
            "nickname": reporter_name_info["nickname"],
            "gender": reporter_name_info["gender"],
            "phone": reporter_phone
        },
        "victims_list": victims_list,
        "victims": {
            "dead": dead,
            "critical": critical,
            "urgent": urgent,
            "safe": safe,
            "child": child,
            "infant": infant
        },
        "items": {
            "firstAid": firstAid,
            "food": food,
            "energy": energy
        }
    }

def generate_synthetic_message(client, style_template, parameters, retries=3):
    """Call Gemini 3.1 Lite to generate disaster messages using style templates and parameters."""
    system_instruction = """You are a creative writer specializing in disaster emergency communication. Your task is to write realistic, natural-sounding social media posts (tweets or Facebook comments in Thai) requesting help during a flood or other disaster in Thailand.

You will be given a set of Ground Truth parameters (names, phones, victims count, items needed, location details) and a real message as a style template.

Your goal is to output the generated Thai message in plain text.

STYLE TEMPLATE INSTRUCTIONS (CRITICAL FOR NATURALNESS):
- The `style_template` is provided ONLY as a reference for: writing style, slang, sentence length, level of urgency, emotional tone, emoji usage, typos, or particles (ครับ/ค่ะ/นะ).
- The factual CONTENT of the `style_template` (specific names, phone numbers, locations, or counts) must be completely IGNORED.
- You MUST write a completely new message that uses the style and tone of the template, but reflects ONLY the facts and data given in the `parameters`. Do NOT carry over any names, phones, or locations from the template.
- If the parameters contain a phone number or location, but the template does not, you MUST still write them in the generated message naturally.
- DO NOT generate formal, polite, or robotic AI-like text (such as "เรียนเจ้าหน้าที่ที่เกี่ยวข้อง ข้าพเจ้าต้องการ..." or "ประกาศขอความช่วยเหลือ..."). The text must read like a real, stressed person typing on a mobile phone during a crisis (short, emotional, slightly chaotic, with emojis, typos, or local Thai terms like เจ๊, เฮีย, กู้ภัย, อาสา).

CRITICAL RULES FOR NATURAL THAI PHRASING:

1. DO NOT write numbers or counts in a literal, robotic, or template-like way.
   - BAD: "บาดเจ็บสาหัส 1 คน, มีเด็ก 2 คน, ต้องการอาหาร 1"
   - GOOD: "แฟนผมโดนไม้ทับขาหักขยับไม่ได้เลยครับ ในบ้านยังมีลูกสาวเล็กๆ อีกสองคน ตอนนี้หิวกันมาก ของกินหมดเกลี้ยงเลย"
2. ZERO-COUNT RULE: If a parameter's count is 0, the generated message MUST NOT mention or imply any details related to that parameter.
   - Example: If `dead` is 0, the message must not mention anyone dying. If `child` is 0, the message must not mention children. If `energy` is 0, the message must not ask for power/flashlights.
3. COORDINATES & MAPS RULE:
   - Location Name: If `location_name` parameter is provided (not null), you MUST integrate that exact location name into the generated message. You can add natural prefixes/suffixes (e.g. "พิกัดซอย...", "ติดอยู่ตรง...") but keep the exact name intact for validation. If `location_name` is null, do NOT mention any location name, landmark, road, or sub-district in the text.
   - Google Map URL: If `google_map_url` is provided (not null), you MUST integrate the Google Map URL into the generated message. If it is null, do NOT include any Google Map URL.
   - Lat/Lng Coordinates: If `lat` and `lng` are provided (not 0.0), you MUST integrate the latitude and longitude coordinate values (e.g. "13.7563, 100.5018" or "พิกัด 20.4272 99.8847") into the message. If they are 0.0, do NOT include any coordinate numbers.
4. Integrate names, nicknames, and phones naturally:
   - Integrate the names exactly as provided in the parameters (including prefixes and last names if present) in a natural way. For example, if name is "คุณสมชาย ใจดี" or "ป้าดา", write it as "คุณสมชาย ใจดี" or "ป้าดา" in the text.
   - Nicknames: If `nickname` is provided (not null) and `use_only_nickname` is false, you can optionally integrate the nickname into the text next to their name (e.g., "ติดต่อ แบงค์ (ปิยะ)", "ป้าสมศรี (ป้าดา)"). If `use_only_nickname` is true, write the nickname (which is already set in their `name` field).
   - For phone numbers, place them as contacts (e.g., "โทรหาพี่แดงได้เลยครับ 081-xxx-xxxx" or "ติดต่อผู้ประสานงาน 089xxxxxxx").
   - If a phone number is specified as "null" or missing, do NOT put any placeholder or phone number for that person.
   - Same Person Scenario: If `contact_victim` and `contact_reporter` have the exact same name, it means the victim is reporting for themselves (first-person report). Write the message from the first-person perspective (e.g., using "ผม", "ฉัน", "หนู") and only mention their name and phone number once in the message (e.g., "ผมชื่อวิน โทร 081-xxx-xxxx ช่วยผมด้วยครับ").
   - Different Persons Scenario: If their names are different, write the message from the reporter's perspective on behalf of the victim (e.g., "ช่วยป้าสมศรีด้วยครับ ติดต่อผมวิชิต โทร 092-xxx-xxxx").
5. Keep the tone realistic: Use exclamation marks, crying emojis (😭, 🙏, 🚨), local abbreviations, spelling variants typical of social media typing under stress, or polite particles like ครับ/ค่ะ/นะคะ.
6. The generated message MUST be in Thai and must closely match the situation details represented by the parameters.
7. VICTIMS CATEGORY GUIDELINES (Aligning with clinical criteria):
   - critical (maps to RED): Describe these victims as trapped (e.g., ติดอยู่บนหลังคา, ดินถล่มทับออกไม่ได้), in severe danger (e.g., น้ำท่วมมิดหัว, กระแสน้ำพัดไป), unresponsive (e.g., หมดสติ, เรียกไม่ตื่น), near-drowning (e.g., จมน้ำ, สำลักน้ำ), or having active severe bleeding (e.g., เลือดไหลพุ่งไม่หยุด).
   - urgent (maps to YELLOW): Describe these victims as injured (e.g., ขาหัก, แขนผิดรูป, กระดูกโผล่), sick (e.g., เป็นไข้สูงซึมมาก, ท้องเสียจนหมดแรง, ทานข้าวไม่ได้อ่อนเพลียมาก), or having moderate difficulty (e.g., หายใจหอบหืด).
   - safe (maps to GREEN): Describe these victims as evacuated, safe, or having minor issues (e.g., แผลถลอกเล็กน้อย) but maybe needing basic food and water.
8. VICTIMS LIST INSTRUCTIONS:
   You will be given a list of individual victims in `victims_list`. For each victim, integrate their details into the generated text using these rules:
   - Name: If `name` is provided (not null), write that exact name. If it is null, refer to them using a generic relation (e.g. "หลานสาว", "แฟนผม", "คนแถวบ้าน").
   - Age Group, Age, and Disclosure:
     - If `age_group` is "child":
7. VICTIMS LIST INSTRUCTIONS:
   - If `age_group` is "child":
     - If `age_disclosure` is "direct" (and age is not null), specify their exact age (e.g., "อายุ 5 ขวบ", "5 ขวบ").
     - If `age_disclosure` is "indirect" (or age is null), refer to them using child-related terms (e.g., "เด็กเล็ก", "ลูกสาวคนเล็ก", "น้อง") without writing the exact age number.
   - If `age_group` is "adult":
     - If `age` is not null and `age_disclosure` is "direct", specify their exact age (e.g., "อายุ 45 ปี", "วัย 45 ปี").
     - If `age` is null or `age_disclosure` is "indirect", refer to them using adult/elderly words (e.g., "คุณยาย", "ลุงข้างบ้าน", "ผู้ป่วยติดเตียง", "แม่ของฉัน") without writing any age numbers.
    - Symptoms & Triage Color:
      For each victim, you MUST integrate the exact Thai text provided in their `symptoms_literal` field into the generated message. Do not translate, change, or paraphrase this symptom text. You must write it exactly as provided in the parameter, but integrate it naturally with prefixes/suffixes (e.g., "ตอนนี้คุณป้า...มีอาการ[symptoms_literal]...", "ติดอยู่ตรง...[symptoms_literal]...").
"""

    user_prompt = json.dumps({
        "style_template": style_template,
        "parameters": parameters
    }, ensure_ascii=False, indent=2)

    backoff = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="google/gemini-3.1-flash-lite",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_tokens=512
            )
            message = response.choices[0].message.content.strip()
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1].strip()
            if message.startswith("'") and message.endswith("'"):
                message = message[1:-1].strip()
            return message
        except Exception as e:
            print(f"Error calling Message Generator (attempt {attempt+1}): {e}")
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0
    return None

def generate_other_message(client, style_template, sub_category, retries=3):
    """Call Gemini 3.1 Lite to generate non-emergency messages (weather warnings, prayers, donations, updates) for classification testing."""
    system_instruction = """You are a creative writer specializing in disaster emergency communication. Your task is to write realistic, natural-sounding social media posts (tweets or Facebook comments in Thai) about a disaster in Thailand that are NOT direct requests for emergency rescue, medical aid, or immediate basic supplies.

You will be given a sub-category of "other" (warning, prayer, donation, update) and a style template.

Your goal is to output the generated Thai message in plain text.

STYLE TEMPLATE INSTRUCTIONS (CRITICAL):
- The `style_template` is provided ONLY as a reference for style, slang, tone, and emoji usage. Ignore its specific facts.
- Generate a new message that strictly belongs to the specified `sub_category`, using the style of the template.
- Avoid robotic or formal language. The post must look like a natural social media post (tweet/Facebook comment) in Thai.

CRITICAL RULES FOR "OTHER" MESSAGES:
1. DO NOT mention any specific individuals needing rescue or medical help in the text.
2. DO NOT write about any injuries, deaths, or critical/trapped victims.
3. The message must fit one of these sub-categories:
   - warning: Weather forecasts, rain alerts, or evacuation advice from authorities (e.g. "ประกาศเตือนภัยจากศูนย์วิจัยน้ำ...").
   - prayer: Messages of moral support, prayers, or expressing sympathy (e.g. "ขอส่งกำลังใจให้เชียงรายปลอดภัยนะครับ").
   - donation: General donation campaigns, relief supply collection, or volunteer recruitment (e.g. "เปิดรับบริจาคสิ่งของจำเป็นเพื่อนำไปแพ็คถุงยังชีพช่วยเหลือ...").
   - update: General updates on water levels, road closures, or weather conditions without active victim reports (e.g. "ระดับน้ำปิงเช้านี้เริ่มลดลงเล็กน้อยแล้วครับ").
4. Keep the tone realistic: Use emojis, abbreviations, or polite particles like ครับ/ค่ะ/นะคะ as appropriate for social media.
5. The generated message MUST be in Thai."""

    user_prompt = json.dumps({
        "style_template": style_template,
        "sub_category": sub_category
    }, ensure_ascii=False, indent=2)

    backoff = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="google/gemini-3.1-flash-lite",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_tokens=512
            )
            message = response.choices[0].message.content.strip()
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1].strip()
            if message.startswith("'") and message.endswith("'"):
                message = message[1:-1].strip()
            return message
        except Exception as e:
            print(f"Error calling Other Message Generator (attempt {attempt+1}): {e}")
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0
    return None

def generate_other_parameters():
    """Generate blank/null ground truth parameters for non-emergency (other) messages."""
    return {
        "location": {
            "location_name": None,
            "google_map_url": None,
            "lat": 0.0,
            "lng": 0.0
        },
        "contact_victim": {
            "name": None,
            "first_name": None,
            "last_name": None,
            "prefix": None,
            "phone": None
        },
        "contact_reporter": {
            "name": None,
            "first_name": None,
            "last_name": None,
            "prefix": None,
            "phone": None
        },
        "victims_list": [],
        "victims": {
            "dead": 0,
            "critical": 0,
            "urgent": 0,
            "safe": 0,
            "child": 0,
            "infant": 0
        },
        "items": {
            "firstAid": 0,
            "food": 0,
            "energy": 0
        }
    }


def validate_generated_text(text, gt_params):
    """Validate generated text against Ground Truth using rule-based checks."""
    if not text:
        return False, "Generated text is empty"
        
    # 1. Validate names
    v_first = gt_params["contact_victim"].get("first_name")
    if v_first and v_first not in text:
        return False, f"Victim first name '{v_first}' missing from text"
    v_last = gt_params["contact_victim"].get("last_name")
    if v_last and v_last not in text:
        return False, f"Victim last name '{v_last}' missing from text"
        
    r_first = gt_params["contact_reporter"].get("first_name")
    if r_first and r_first not in text:
        return False, f"Reporter first name '{r_first}' missing from text"
    r_last = gt_params["contact_reporter"].get("last_name")
    if r_last and r_last not in text:
        return False, f"Reporter last name '{r_last}' missing from text"
        
    # 2. Validate phones
    v_phone = gt_params["contact_victim"]["phone"]
    if v_phone:
        cleaned_target = clean_phone(v_phone)
        cleaned_text = clean_phone(text)
        if cleaned_target not in cleaned_text:
            return False, f"Victim phone '{v_phone}' missing from text"
            
    r_phone = gt_params["contact_reporter"]["phone"]
    if r_phone:
        cleaned_target = clean_phone(r_phone)
        cleaned_text = clean_phone(text)
        if cleaned_target not in cleaned_text:
            return False, f"Reporter phone '{r_phone}' missing from text"
            
    # 3. Validate Location
    loc_name = gt_params["location"]["location_name"]
    if loc_name and loc_name not in text:
        return False, f"Location name '{loc_name}' missing from text"
        
    # 4. Validate Google Map URL
    map_url = gt_params["location"]["google_map_url"]
    if map_url and map_url not in text:
        return False, f"Google Map URL '{map_url}' missing from text"
        
    # 5. Validate coordinates
    lat = gt_params["location"]["lat"]
    lng = gt_params["location"]["lng"]
    if lat != 0.0 and lng != 0.0:
        lat_check = f"{lat:.3f}"[:-1]
        lng_check = f"{lng:.3f}"[:-1]
        if lat_check not in text or lng_check not in text:
            if f"{lat}" not in text or f"{lng}" not in text:
                return False, f"Coordinates ({lat}, {lng}) missing from text"
                
    # 6. Validate symptoms_literal
    for victim in gt_params.get("victims_list", []):
        symptom = victim.get("symptoms_literal")
        if symptom and symptom not in text:
            return False, f"Symptom '{symptom}' missing from text"
            
    return True, ""

def process_single_message(style_template, parameters):
    """Generate and validate a single message, retrying up to 3 times if validation fails."""
    for attempt in range(3):
        text = generate_synthetic_message(openrouter_client, style_template, parameters)
        is_valid, reason = validate_generated_text(text, parameters)
        if is_valid:
            return text, attempt + 1
        else:
            print(f"Validation failed (Attempt {attempt+1}): {reason}. Retrying...")
            
    # Fallback to last generated text if all attempts fail
    return text, 3

def main():
    print("Loading style templates from merged_clean.csv...")
    merged_clean_path = "e:/nlp-for-disaster/dataset/clean/merged_clean.csv"
    if not os.path.exists(merged_clean_path):
        raise FileNotFoundError(f"merged_clean.csv not found at {merged_clean_path}!")
        
    df_clean = pd.read_csv(merged_clean_path)
    templates = df_clean.to_dict('records')
    print(f"Loaded {len(templates)} style templates.")
    
    import sys
    num_samples = 2000
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            pass
    elif os.getenv("NUM_SAMPLES"):
        try:
            num_samples = int(os.getenv("NUM_SAMPLES"))
        except ValueError:
            pass
    print(f"Generating {num_samples} samples...")
    print(f"Pre-generating location pool of size {num_samples}...")
    location_pool = generate_location_pool(typhoon_client, num_needed=num_samples)
    
    output_rows = []
    output_path = "e:/nlp-for-disaster/dataset/clean/synthetic_ner_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load already generated rows if file exists to support resuming
    completed_ids = set()
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            output_rows = df_existing.to_dict('records')
            completed_ids = set(df_existing['synthetic_id'].tolist())
            print(f"Resuming generation. Loaded {len(completed_ids)} existing rows.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            output_rows = []
            
    print(f"Starting synthetic dataset generation pipeline...")
    
    start_time = time.time()
    for i in range(len(completed_ids) + 1, num_samples + 1):
        synth_id = f"SYN_NER_{i:03d}"
        print(f"\n--- Generating row {i}/{num_samples} ({synth_id}) ---")
        
        # Pick random template
        template_row = random.choice(templates)
        style_template = template_row['text']
        source_id = template_row['id']
        
        # Determine classification category (50% help_request, 50% other)
        classification_category = "help_request" if random.random() < 0.5 else "other"
        
        if classification_category == "help_request":
            # Generate random parameters for help request
            params = generate_random_parameters(location_pool)
            # Generate text and validate
            text, attempts = process_single_message(style_template, params)
        else:
            # Generate blank parameters for other category
            params = generate_other_parameters()
            # Generate other category message (weather warnings, prayers, donations, updates)
            sub_cat = random.choice(["warning", "prayer", "donation", "update"])
            text = generate_other_message(openrouter_client, style_template, sub_cat)
            
        # Prepare row
        row = {
            "synthetic_id": synth_id,
            "generated_text": text,
            "gt_is_help_request": (classification_category == "help_request"),
            "gt_classification_category": classification_category,
            "gt_location_name": params["location"]["location_name"],
            "gt_google_map_url": params["location"]["google_map_url"],
            "gt_lat": params["location"]["lat"],
            "gt_lng": params["location"]["lng"],
            "gt_victim_name": params["contact_victim"]["name"],
            "gt_victim_phone": params["contact_victim"]["phone"],
            "gt_victim_gender": params["contact_victim"].get("gender"),
            "gt_victim_nickname": params["contact_victim"].get("nickname"),
            "gt_reporter_name": params["contact_reporter"]["name"],
            "gt_reporter_phone": params["contact_reporter"]["phone"],
            "gt_reporter_gender": params["contact_reporter"].get("gender"),
            "gt_reporter_nickname": params["contact_reporter"].get("nickname"),
            "gt_victims_json": json.dumps(params["victims_list"], ensure_ascii=False),
            "gt_dead": params["victims"]["dead"],
            "gt_critical": params["victims"]["critical"],
            "gt_urgent": params["victims"]["urgent"],
            "gt_safe": params["victims"]["safe"],
            "gt_child": params["victims"]["child"],
            "gt_infant": params["victims"]["infant"],
            "gt_item_firstaid": params["items"]["firstAid"],
            "gt_item_food": params["items"]["food"],
            "gt_item_energy": params["items"]["energy"],
            "source_template_id": source_id
        }
        
        output_rows.append(row)
        
        # Save periodically to prevent data loss
        if i % 10 == 0 or i == num_samples:
            df_out = pd.DataFrame(output_rows)
            df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Successfully saved progress to {output_path} ({len(output_rows)} rows).")
            
        # Optional: Sleep slightly to be polite to APIs
        time.sleep(0.2)
        
    end_time = time.time()
    print(f"\nSynthetic NER dataset generation complete!")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds.")
    print(f"Final output file saved to {output_path}")

if __name__ == "__main__":
    main()
