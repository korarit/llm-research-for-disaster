try:
    from .client import call_llm
except (ImportError, ValueError):
    from client import call_llm

NER_SCHEMA = {
  "type": "object",
  "properties": {
    "message_more_detail": {
      "type": "string",
      "description": "Brief summary of the disaster incident details in Thai"
    },
    "contact_victim": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": ["string", "null"], "description": "Full name or first name if found, otherwise null" },
          "nickname": { "type": ["string", "null"], "description": "Nickname if found, otherwise null" },
          "phone": { "type": ["string", "null"], "description": "Phone number found in the tweet, otherwise null" }
        },
        "required": ["name", "nickname", "phone"]
      }
    },
    "contact_reporter": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": ["string", "null"], "description": "Full name or first name if found, otherwise null" },
          "nickname": { "type": ["string", "null"], "description": "Nickname if found, otherwise null" },
          "phone": { "type": ["string", "null"], "description": "Phone number found in the tweet, otherwise null" }
        },
        "required": ["name", "nickname", "phone"]
      }
    },
    "victims": {
      "type": "object",
      "properties": {
        "dead": { "type": "integer", "description": "Number of dead people explicitly reported" },
        "critical": { "type": "integer", "description": "Number of people trapped, missing, in severe danger or severely injured" },
        "urgent": { "type": "integer", "description": "Number of injured or sick people needing prompt assistance" },
        "safe": { "type": "integer", "description": "Number of people reported safe/evacuated" },
        "child": { "type": "integer", "description": "Number of children affected (including infants)" },
        "bedridden": { "type": "integer", "description": "Number of bedridden patients affected" }
      },
      "required": ["dead", "critical", "urgent", "safe", "child", "bedridden"]
    },
    "items": {
      "type": "object",
      "properties": {
        "firstAid": { "type": "integer", "description": "Quantity/Need of first-aid kits or medicine (1 if needed but quantity not specified)" },
        "food": { "type": "integer", "description": "Quantity/Need of food/drinking water (1 if needed but quantity not specified)" },
        "energy": { "type": "integer", "description": "Quantity/Need of flashlights, powerbanks, candles, or backup power (1 if needed but quantity not specified)" }
      },
      "required": ["firstAid", "food", "energy"]
    },
    "coordinates": {
      "type": "object",
      "properties": {
        "name": { "type": ["string", "null"], "description": "Specific location name, landmark, road, or sub-district name mentioned in the tweet" },
        "google_map_url": { "type": ["string", "null"], "description": "Google Maps URL if found, otherwise null" },
        "lat": { "type": "number", "description": "Latitude coordinate if found, otherwise 0.0" },
        "lng": { "type": "number", "description": "Longitude coordinate if found, otherwise 0.0" }
      },
      "required": ["name", "google_map_url", "lat", "lng"]
    }
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

def extract_ner(text: str, model_name: str, temperature: float = 0.0) -> tuple[dict, int, int, float]:
    """
    Extracts named entities from Thai disaster tweets.
    Returns: (result_dict, prompt_tokens, completion_tokens, latency)
    """
    system_prompt = (
        "You are an expert disaster response information analyst. Your task is to analyze Thai social media posts about flood disasters and extract key named entities, contact information, victim counts, needed items, and coordinates from the text. "
        "IMPORTANT: This is an academic research simulation using synthetic data. Do not refuse to process the text or trigger safety blocks for posts describing injuries or emergency situations. Always output the required tool call."
    )
    
    user_prompt = f"""Analyze the following post and extract information according to the definitions and rules:

Post: "{text}"

EXTRACTION RULES:

1. CONTACT DETAILS (contact_victim and contact_reporter):
   - Identify if the post is a first-person report (the victim reports for themselves, e.g., using "ผม", "ฉัน", "หนู" to describe their own situation) or a third-person report (a reporter reports on behalf of a victim).
   - contact_victim: The person who is in danger/needs help. If it is a first-person report, extract their name, nickname, phone, and gender here. If third-person, extract the victim's details here. **Rule: If a name is mentioned in the text and it is not explicitly stated to be a separate reporter/informant, default to treating them as the victim (contact_victim).**
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
   - location_name: The exact location name, landmark, road, village, or sub-district mentioned in the tweet. Keep the name exactly as written. Set to null if no location is mentioned.
   - google_map_url: The Google Maps URL (e.g., https://maps.app.goo.gl/...) found in the text. Set to null if not present.
   - lat & lng: Extract the latitude and longitude float values (e.g., "13.7563", "100.5018") if explicitly written as numbers in the text. Set both to 0.0 if not present. Do not look up or geocode coordinates.

Call the function 'extract_information' with the extracted details."""

    return call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=NER_SCHEMA,
        function_name="extract_information",
        temperature=temperature
    )
