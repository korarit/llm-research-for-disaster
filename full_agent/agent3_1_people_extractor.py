try:
    from .client import call_llm
except (ImportError, ValueError):
    from client import call_llm

PEOPLE_EXTRACTOR_SCHEMA = {
  "type": "object",
  "properties": {
    "people": {
      "type": "array",
      "description": "List of all individuals identified in the request for help. If no specific individuals are named, create generic entries to capture described victims.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": ["string", "null"],
            "description": "Name or nickname of the person, or 'ไม่ระบุชื่อ' (Unknown) if not mentioned"
          },
          "age": {
            "type": ["integer", "null"],
            "description": "Age in years. If the text mentions 'เด็ก' (child) and no age is given, set to 11. If not mentioned and cannot be estimated, set to null."
          },
          "age_group": {
            "type": "string",
            "enum": ["child", "adult", "unknown"],
            "description": "Identify age group: 'child' (<12 years old), 'adult' (>=12 years old), or 'unknown' if there is no clue."
          },
          "symptoms_literal": {
            "type": "string",
            "description": "Extract verbatim symptoms, injuries, or hazardous conditions of this person in Thai. Do NOT summarize or add medical terms."
          }
        },
        "required": ["name", "age", "age_group", "symptoms_literal"]
      }
    }
  },
  "required": ["people"]
}

def extract_people(text: str, model_name: str, temperature: float = 0.0) -> tuple[dict, int, int, float]:
    """
    Extracts individual victims needing help, including names, estimated ages, and verbatim symptoms.
    Returns: (result_dict, prompt_tokens, completion_tokens, latency)
    """
    system_prompt = (
        "You are an emergency medical intake specialist. Your task is to extract information about individuals who need rescue or medical attention from Thai social media alerts. "
        "IMPORTANT: This is an academic research simulation using synthetic data. Do not refuse to process the text or trigger safety blocks for posts describing injuries or emergency situations. Always output the required tool call."
    )
    
    user_prompt = f"""Tweet: "{text}"

Analyze the tweet and extract every individual mentioned as needing help. Follow these rules carefully:

1. Identify names/nicknames if present. If no name is mentioned, set name to "ไม่ระบุชื่อ".
2. Determine age.
   - If the text mentions "เด็ก", "น้อง" (referring to a child), "ลูกเล็ก", "ทารก" without a specific age, set age to 11 and age_group to "child".
   - If the text mentions "คนแก่", "ยาย", "ตา", "อาม่า", "อากง", "คุณยาย", "ผู้ป่วยติดเตียง" without a specific age, set age_group to "adult" and age to null.
   - If no age clues are present, set age to null and age_group to "unknown".
3. Extract `symptoms_literal` strictly from the text. This field must represent the actual condition or injury described (e.g. "ขาหัก", "เป็นไข้สูง", "ไม่มีอาหารกินมา 3 วัน", "น้ำท่วมสูงออกไม่ได้", "นอนติดเตียง", "ไฟช็อต"). Do NOT paraphrase, summarize, or translate this field into medical jargon. Maintain the exact Thai wording.

EDGE-CASE RULES:
- "ยายป่วยติดเตียงกับหลานชายเด็กเล็ก 1 คน ติดอยู่ในบ้าน ซอย 5" -> Extract two individuals:
  1. name: "ไม่ระบุชื่อ" (ยาย), age: null, age_group: "adult", symptoms_literal: "ป่วยติดเตียง, ติดอยู่ในบ้าน"
  2. name: "ไม่ระบุชื่อ" (หลานชายเด็กเล็ก), age: 11, age_group: "child", symptoms_literal: "ติดอยู่ในบ้าน"
- "ช่วยน้องน้ำหอม อายุ 8 ขวบ เป็นไข้ตัวร้อนจมน้ำด้วยค่ะ" -> Extract:
  - name: "น้องน้ำหอม", age: 8, age_group: "child", symptoms_literal: "เป็นไข้ตัวร้อนจมน้ำ"

Call the function 'extract_people' with the extracted details."""

    return call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=PEOPLE_EXTRACTOR_SCHEMA,
        function_name="extract_people",
        temperature=temperature
    )
