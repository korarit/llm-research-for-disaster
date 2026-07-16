# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Disaster Alert Labeling Experiment - 04)

เอกสารนี้รวบรวมข้อสรุปแนวทาง โครงสร้าง## 1. โครงสร้างการสกัดชื่อจำเพาะ (Named Entity Recognition - NER)

ในการสกัด Entity จากข้อความภัยพิบัติภาษาไทยของชุดข้อมูลสังเคราะห์ (Synthetic Dataset) จะมุ่งเน้นไปที่ข้อมูลหลัก 4 ส่วนที่ถูกสร้างขึ้นโดยเครื่องมือจำลองสถานการณ์:
1. **Location & Coordinates (สถานที่และพิกัด)**: สกัดชื่อสถานที่เกิดเหตุ (ถนน, ตำบล, อำเภอ, จังหวัด, จุดสังเกต), Google Map URL และพิกัดทางภูมิศาสตร์ (Latitude, Longitude) ที่ระบุในข้อความโดยตรง
2. **Contact Details (ข้อมูลติดต่อ)**: สกัดชื่อ (Name), ชื่อเล่น (Nickname), เบอร์โทรศัพท์ (Phone) และเพศ (Gender) ของผู้ประสบภัยหลัก (contact_victim) และผู้แจ้งเหตุ (contact_reporter)
3. **Victims Count (จำนวนผู้ประสบภัย)**: สกัดจำนวนผู้เสียชีวิต (dead), ผู้ประสบภัยวิกฤต (critical), ผู้ประสบภัยเร่งด่วน (urgent), ผู้ประสบภัยที่ปลอดภัย (safe), จำนวนเด็ก (child - รวมทารกด้วย) และผู้ป่วยติดเตียง (bedridden)
4. **Items Needed (สิ่งของที่จำเป็น)**: สกัดจำนวน/ความต้องการอุปกรณ์ปฐมพยาบาล/ยา (firstAid), อาหาร/น้ำดื่ม (food) และระบบไฟฟ้าสำรอง/ไฟฉาย/พาวเวอร์แบงค์ (energy)

*หมายเหตุ: ในการทดลอง Experiment 04 นี้ จะไม่มีการสกัดข้อมูลระดับความรุนแรง (Severity) หรือระดับประเภทภัยพิบัติอื่น ๆ*

---

## 2. การกำหนดค่าโมเดลและสถาปัตยกรรมระบบ (Model & Pipeline Architecture)

ระบบการทดลองจะประยุกต์ใช้โมดูลและไลบรารีในลักษณะเดิมเพื่อทำการทดสอบโมเดล MoE ทั้ง 3 รุ่นร่วมกับชุดข้อมูลภัยพิบัติภาษาไทยสังเคราะห์:

### 2.1 การเชื่อมต่อโมเดลผ่าน API
การเชื่อมต่อโมเดลทุกตัวจะใช้การเรียก API ภายนอก (External API Call) ทั้งหมดผ่าน OpenAI-compatible client SDK โดยอ้างอิง Endpoint และคีย์ผู้ให้บริการดังนี้:

1. **Gemma 4** (เข้าถึงผ่าน OpenRouter API):
   - **โมเดลคีย์:** `google/gemma-4-26b-a4b-it`
   - ** Endpoint:** `https://openrouter.ai/api/v1`
   - ** API Key:** `os.environ.get("OPENROUTER_API_KEY")`

2. **deepseek-v4-flash** (เข้าถึงผ่าน OpenRouter API):
   - **โมเดลคีย์:** `deepseek/deepseek-v4-flash`
   - ** Endpoint:** `https://openrouter.ai/api/v1`
   - ** API Key:** `os.environ.get("OPENROUTER_API_KEY")`

3. **typhoon-v2.5** (เข้าถึงผ่าน Typhoon API):
   - **โมเดลคีย์:** `typhoon-v2.5-30b-a3b-instruct`
   - ** Endpoint:** `https://api.opn.ai/v1`
   - ** API Key:** `os.environ.get("TYPHOON_API_KEY")`

### 2.2 รูปแบบฟังก์ชันเรียกใช้งาน (Function Calling Schema)

ในการประมวลผลข้อมูลผ่าน API จะใช้คุณลักษณะการเรียกฟังก์ชัน (Function Calling) โดยแบ่งการประมวลผลเป็น 2 ขั้นตอน (2-Stage Pipeline):
- **Stage 1 (Help Request Filtering)**: คัดแยกข้อความว่าต้องการขอความช่วยเหลือฉุกเฉิน (help_request) หรือเป็นข้อความประเภทอื่น (other เช่น เตือนภัยทั่วไป, ให้กำลังใจ/สวดมนต์, ขอรับบริจาคทั่วไป, อัปเดตสถานการณ์น้ำ)
- **Stage 2 (NER Extraction)**: ดึงข้อมูล NER และตัวนับทางคลินิก/สิ่งของที่จำเป็นสำหรับข้อความที่ผ่านตัวกรองใน Stage 1

กำหนดโครงสร้างข้อมูลสำหรับฟังก์ชันเรียกใช้งานในรูปแบบ Pydantic คลาส ดังนี้:

```python
from pydantic import BaseModel, Field
from typing import Optional

# โครงสร้างสำหรับ Stage 1: Help Request Filtering
class Stage1Result(BaseModel):
    is_help_request: bool = Field(
        description="True if the message is a direct request for emergency rescue, medical aid, or immediate basic needs (help_request). False if it is a general update, weather warning, prayer, general donation campaign, or other non-emergency content (other)."
    )

# โครงสร้างย่อยสำหรับ Stage 2: NER Extraction
class ContactDetail(BaseModel):
    name: Optional[str] = Field(
        description="Full name, first name, prefix + name, or nickname of the person. Set to null if not mentioned."
    )
    nickname: Optional[str] = Field(
        description="Nickname of the contact person (e.g., แบงค์, ส้ม, ป้าดา) if explicitly mentioned. Set to null if not mentioned."
    )
    phone: Optional[str] = Field(
        description="Thai mobile phone number found for this person (e.g. 0812345678, 089-123-4567). Keep spelling exactly as written. Set to null if not mentioned."
    )
    gender: Optional[str] = Field(
        description="Gender of the contact ('male' or 'female') inferred from prefix, nicknames, pronouns, or name. Set to null if cannot be determined."
    )

class VictimsCount(BaseModel):
    dead: int = Field(
        default=0, 
        description="Number of dead people explicitly reported"
    )
    critical: int = Field(
        default=0, 
        description="Number of people trapped, missing, in severe danger (e.g., RED triage: trapped on roof, landslide, swept away, unconscious, near-drowning, severe bleeding)"
    )
    urgent: int = Field(
        default=0, 
        description="Number of injured or sick people needing prompt assistance (e.g., YELLOW triage: bone fracture, high fever, severe diarrhea/vomiting, breathing difficulty)"
    )
    safe: int = Field(
        default=0, 
        description="Number of people reported safe/evacuated (e.g., GREEN triage: safe, evacuated, minor scratches)"
    )
    child: int = Field(
        default=0, 
        description="Number of children affected (age <= 11 or referred to as child/kid/น้อง/เด็ก/ทารก)"
    )
    bedridden: int = Field(
        default=0, 
        description="Number of bedridden patients (ผู้ป่วยติดเตียง, นอนติดเตียง, ป่วยติดเตียง)"
    )

class ItemsCount(BaseModel):
    firstAid: int = Field(
        default=0, 
        description="Quantity/Need of first-aid kits or medicine (ยารักษาโรค, ยา, ชุดปฐมพยาบาล). Set to quantity needed, or 1 if needed but quantity is not specified."
    )
    food: int = Field(
        default=0, 
        description="Quantity/Need of food/drinking water (อาหาร, น้ำดื่ม, ข้าวกล่อง, ของกิน). Set to quantity needed, or 1 if needed but quantity is not specified."
    )
    energy: int = Field(
        default=0, 
        description="Quantity/Need of flashlights, powerbanks, candles, or backup power (ไฟสำรอง, แบตสำรอง, พาวเวอร์แบงค์, ไฟฉาย, เทียน, เครื่องปั่นไฟ). Set to quantity needed, or 1 if needed but quantity is not specified."
    )

class CoordinatesDetail(BaseModel):
    location_name: Optional[str] = Field(
        description="Specific location name, landmark, road, or sub-district name mentioned in the text. Set to null if not mentioned."
    )
    google_map_url: Optional[str] = Field(
        default=None, 
        description="Google Map URL (e.g., https://maps.app.goo.gl/...) found in the text. Set to null if not mentioned."
    )
    lat: float = Field(
        default=0.0, 
        description="Latitude coordinate (e.g., 13.7563) found in the text. Set to 0.0 if not mentioned."
    )
    lng: float = Field(
        default=0.0, 
        description="Longitude coordinate (e.g., 100.5018) found in the text. Set to 0.0 if not mentioned."
    )

class NERResult(BaseModel):
    message_more_detail: str = Field(
        description="Brief summary of the disaster incident details in Thai"
    )
    contact_victim: Optional[ContactDetail] = Field(
        description="Contact details of the victim. Set to null if not mentioned. If the victim is reporting for themselves (first-person), this should contain their details."
    )
    contact_reporter: Optional[ContactDetail] = Field(
        description="Contact details of the reporter/informant who is reporting on behalf of the victim. Set to null if not mentioned. If first-person report, this should be the same as contact_victim."
    )
    victims: VictimsCount
    items: ItemsCount
    coordinates: CoordinatesDetail
```

### 2.3 ข้อกำหนดเฉพาะในการทดลอง
1. **สกัดข้อมูลจากข้อความต้นฉบับเท่านั้น**: ห้ามใช้การเรียกโมเดลอื่นเสริมภายนอก (เช่น API ค้นหาภูมิศาสตร์ภายนอก) ให้สกัดพิกัด `lat`, `lng` และ `google_map_url` เฉพาะในกรณีที่เขียนระบุไว้ในข้อความโซเชียลมีเดียภาษาไทยเท่านั้น หากไม่มีระบุให้กำหนดค่าเริ่มต้นเป็น `0.0` และ `null` ตามลำดับ
2. **ไม่ใช้ชุดข้อมูลในการเขียน Prompt**: ห้ามนำรายชื่อจังหวัด อำเภอ รายชื่อคน หรือชุดอาการจริงจากไฟล์ dataset มาเขียนเป็นคีย์เวิร์ดแบบ Hardcode ลงใน Prompt ของเอเจนต์เด็ดขาด (แต่ให้เขียนกฎคำอธิบายนิยามประเภทบุคคล เพศ และตัวนับทางคลินิกเป็นภาษาอังกฤษทั่วไป)
3. **เรื่องความรุนแรง**: การทดลองนี้จะไม่มีการจัดระดับความรุนแรงของภัยพิบัติใด ๆ (ไม่มีฟิลด์ Severity)

---

## 3. การออกแบบคำสั่ง (Prompt Design Template)

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะเขียนเป็นภาษาอังกฤษเพื่อประสิทธิภาพสูงสุดของโมเดล MoE และความคุ้มค่าด้าน Token (Token Cost Efficiency) โดยกำหนดโครงสร้างของคำสั่งดังนี้:

### 3.1 Stage 1: Help Request Filtering Prompt
- **Function Name**: `filter_help_request`
- **System Instruction**:
  ```markdown
  You are an expert disaster response analyst. Your task is to analyze social media posts (tweets or Facebook comments in Thai) and determine if they contain a direct request for emergency rescue, medical aid, or immediate assistance (such as food/water/power) for specific victims in danger.
  ```
- **User Prompt Template**:
  ```markdown
  Analyze the following post and determine if it is a direct help request:
  
  Post: "{text}"
  
  CLASSIFICATION RULES:
  - set is_help_request to True (help_request) if the post contains an active report of victims needing rescue, medical aid, or immediate basic supplies (food, clean water, first aid, power).
  - set is_help_request to False (other) if the post is:
    - A general weather warning, rain warning, or evacuation announcement from authorities without reporting active trapped/injured victims.
    - A post of moral support, prayers, or expressing condolences (e.g., "ขอให้ปลอดภัย", "ส่งกำลังใจให้").
    - A general donation campaign, relief supply collection, or volunteer recruitment (e.g., "เปิดรับบริจาค", "รับบริจาค").
    - A general situational update on water levels, road status, or weather without active victim rescue reports.
  
  Call the function 'filter_help_request' with your decision.
  ```

### 3.2 Stage 2: NER Extraction Prompt
- **Function Name**: `extract_information`
- **System Instruction**:
  ```markdown
  You are an expert disaster response information analyst. Your task is to analyze Thai social media posts about flood disasters and extract key named entities, contact information, victim counts, needed items, and coordinates from the text.
  ```
- **User Prompt Template**:
  ```markdown
  Analyze the following post and extract information according to the definitions and rules:
  
  Post: "{text}"
  
  EXTRACTION RULES:
  
  1. CONTACT DETAILS (contact_victim and contact_reporter):
     - Identify if the post is a first-person report (the victim reports for themselves, e.g., using "ผม", "ฉัน", "หนู" to describe their own situation) or a third-person report (a reporter reports on behalf of a victim).
     - contact_victim: The person who is in danger/needs help. If it is a first-person report, extract their name, nickname, phone, and gender here. If third-person, extract the victim's details here.
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
     - location_name: The exact location name, landmark, road, village, or sub-district mentioned in the text. Keep the name exactly as written. Set to null if no location is mentioned.
     - google_map_url: The Google Maps URL (e.g., https://maps.app.goo.gl/...) found in the text. Set to null if not present.
     - lat & lng: Extract the latitude and longitude float values (e.g., "13.7563", "100.5018") if explicitly written as numbers in the text. Set both to 0.0 if not present. Do not look up or geocode coordinates.
  
  Call the function 'extract_information' with the extracted details.
  ```

---

## 4. แหล่งอ้างอิงและที่มา (References & Sources)

1. **สคริปต์สกัดชุดข้อมูลสังเคราะห์ภาษาไทย**
   - **ซอร์สโค้ด:** [generate_synthetic_ner.py](file:///e:/nlp-for-disaster/generate_synthetic_ner.py)
   - สคริปต์นี้สร้างชุดข้อมูลจำลองความยาว 2,000 ข้อความ (`synthetic_ner_dataset.csv`) โดยดึงข้อมูล Ground Truth ของชื่อ เพศ เบอร์โทรศัพท์ สถานที่ พิกัด แผนที่ ตลอดจนจำนวนผู้ประสบภัยระดับ RED/YELLOW/GREEN และชนิดสิ่งของที่จำเป็น

2. **งานวิจัยที่เกี่ยวข้องและแนวคิดหลัก**
   - **ชื่องานวิจัย**: [Zero-Shot Social Media Crisis Classification: A Training-Free Multimodal Approach](https://digibug.ugr.es/bitstream/handle/10481/111587/applsci-16-02192.pdf?sequence=1&isAllowed=y) (MDPI Applied Sciences, 2026)
   - **ซอร์สโค้ดอ้างอิงของงานวิจัย (Local Workspace)**: [disaster_classification.ipynb](file:///e:/nlp-for-disaster/ref/disaster_classification.ipynb)

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จากชุดข้อมูลสังเคราะห์ทั้งหมด 2,000 รายการของ Experiment 04 จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:

```text
e:/nlp-for-disaster/exp4/results/
├── deepseek-v4-flash_results.csv        <- บันทึกผลการวิเคราะห์และโครงสร้าง NER ที่สกัดได้
├── typhoon-v2.5_results.csv             <- บันทึกผลการวิเคราะห์และโครงสร้าง NER ที่สกัดได้
├── gemma-4_results.csv                  <- บันทึกผลการวิเคราะห์และโครงสร้าง NER ที่สกัดได้
└── model_comparison_metrics.csv         <- สรุปเปรียบเทียบผลลัพธ์การคัดแยกประเภทและ NER
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema)

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `synthetic_id` | ไอดีข้อความทวีตสังเคราะห์ (อิงจากชุดข้อมูลนำเข้า) | `SYN_NER_001` |
| `generated_text` | ข้อความโซเชียลมีเดียภาษาไทยที่สังเคราะห์ขึ้น | *“ลุงสมชาย ใจดี ขาหักขยับไม่ได้... โทร 081-234-5678”* |
| `pred_is_help_request` | ผลการคัดแยก Stage 1 (True / False) | `True` |
| `pred_location_name` | ผลสกัดชื่อสถานที่ (`coordinates.location_name`) | `ซอย 4 เหมืองแดง แม่สาย เชียงราย` |
| `pred_google_map_url` | ผลสกัดลิงก์แผนที่ (`coordinates.google_map_url`) | `https://maps.app.goo.gl/abcdefg` |
| `pred_lat` | ผลสกัดพิกัดละติจูด (`coordinates.lat`) | `20.4272` |
| `pred_lng` | ผลสกัดพิกัดลองจิจูด (`coordinates.lng`) | `99.8847` |
| `pred_victim_name` | ผลสกัดชื่อผู้ประสบภัยหลัก (`contact_victim.name`) | `ลุงสมชาย ใจดี` |
| `pred_victim_nickname` | ผลสกัดชื่อเล่นผู้ประสบภัยหลัก (`contact_victim.nickname`) | `สมชาย` |
| `pred_victim_phone` | ผลสกัดเบอร์โทรผู้ประสบภัยหลัก (`contact_victim.phone`) | `081-234-5678` |
| `pred_victim_gender` | เพศที่สกัดได้ของผู้ประสบภัยหลัก (`contact_victim.gender`) | `male` |
| `pred_reporter_name` | ผลสกัดชื่อผู้แจ้งเรื่อง (`contact_reporter.name`) | `ลุงสมชาย ใจดี` |
| `pred_reporter_nickname` | ผลสกัดชื่อเล่นผู้แจ้งเรื่อง (`contact_reporter.nickname`) | `สมชาย` |
| `pred_reporter_phone` | ผลสกัดเบอร์โทรผู้แจ้งเรื่อง (`contact_reporter.phone`) | `081-234-5678` |
| `pred_reporter_gender` | เพศที่สกัดได้ของผู้แจ้งเรื่อง (`contact_reporter.gender`) | `male` |
| `pred_dead` | จำนวนผู้เสียชีวิตที่สกัดได้ (`victims.dead`) | `0` |
| `pred_critical` | จำนวนเคสวิกฤต/สีแดงที่สกัดได้ (`victims.critical`) | `0` |
| `pred_urgent` | จำนวนเคสเร่งด่วน/สีเหลืองที่สกัดได้ (`victims.urgent`) | `1` |
| `pred_safe` | จำนวนเคสปลอดภัย/สีเขียวที่สกัดได้ (`victims.safe`) | `0` |
| `pred_child` | จำนวนเด็กที่สกัดได้ (`victims.child`) | `0` |
| `pred_bedridden` | จำนวนผู้ป่วยติดเตียงที่สกัดได้ (`victims.bedridden`) | `0` |
| `pred_item_firstaid` | จำนวนความต้องการชุดปฐมพยาบาล (`items.firstAid`) | `1` |
| `pred_item_food` | จำนวนความต้องการอาหารและน้ำ (`items.food`) | `0` |
| `pred_item_energy` | จำนวนความต้องการพลังงาน/ไฟฉาย (`items.energy`) | `0` |
| `latency_seconds` | ระยะเวลาที่ใช้ประมวลผลสะสมในสอง Stage (วินาที) | `1.5` |
| `token_in_use` | จำนวน Token ขาเข้าสะสมที่ใช้ประมวลผลเอเจนต์ | `520` |
| `token_out_use` | จำนวน Token ขาออกสะสมที่ใช้ประมวลผลเอเจนต์ | `120` |

### 5.2 การคำนวณมาตรวัดความแม่นยำ (Evaluation Metrics)

การวัดความถูกต้องจะถูกวิเคราะห์แยกส่วนตามประเภทข้อมูล ดังนี้:

1. **Stage 1 (Classification)**:
   - ตรวจสอบ `pred_is_help_request` เทียบกับ `gt_is_help_request` (ข้อมูลทุกแถว 2,000 แถว)
   - คำนวณหาค่า: **Accuracy**, **Precision**, **Recall**, และ **F1-Score**

2. **Stage 2 (NER Entity Extraction)**:
   (จะถูกประเมินเฉพาะในแถวที่เฉลยจริง `gt_is_help_request == True` เท่านั้น)
   - **ข้อมูลเบอร์โทรศัพท์ (Phone Numbers)**:
     - นำข้อความที่สกัดได้มาตัดอักขระที่ไม่ใช่ตัวเลขออกให้เหลือเฉพาะตัวเลข (เช่น `081-234-5678` -> `0812345678`)
     - ทำการประเมินเทียบ Ground Truth แบบ **Exact Match (EM)**
   - **ข้อมูลข้อความทั่วไป (Names, Nicknames, Location Name, Google Map URL)**:
     - ทำความสะอาดข้อมูลโดยตัดช่องว่าง (Whitespace) หัวท้าย และปรับอักษรภาษาอังกฤษเป็นตัวเล็ก (Lowercase)
     - เปรียบเทียบแบบ **Exact Match (EM)** โดยกรณีที่ไม่มีระบุทั้งคู่ (มีค่าเป็น `null` หรือ `None`) ให้นับเป็น EM = 1
   - **ข้อมูลเพศ (Gender)**:
     - เปรียบเทียบค่าตรงกันตามข้อความ `male`, `female` หรือ `null` แบบ Exact Match
   - **ข้อมูลพิกัดละติจูด/ลองจิจูด (Coordinates)**:
     - ตรวจสอบค่าพิกัด `lat`, `lng` แบบค่าสัมบูรณ์ผลต่างความคลาดเคลื่อน (Absolute Difference Tolerance)
     - โดยการทำนายจะถือว่าถูกต้องหาก `abs(pred_coord - gt_coord) < 0.001` (หรือพิกัดเป็น 0.0 ทั้งคู่)
   - **ข้อมูลจำนวนนับตัวเลข (Victims and Items Counts)**:
     - ประเมินผลแยกรายฟิลด์ทั้ง 9 ฟิลด์ (dead, critical, urgent, safe, child, bedridden, firstaid, food, energy)
     - คำนวณหาค่า: **Mean Absolute Error (MAE)** และอัตราความถูกต้องตรงกันแบบ **Exact Match Rate** ในแต่ละฟิลด์
นาดภาพให้เท่ากันที่ `512x512 px` (JPEG, Quality 85%) แล้วเปลี่ยนไฟล์รูปเป็น Base64 String เพื่อจัดชุดคำสั่งแบบ Multimodal (Text + Image) ส่งตรงผ่าน API
     3. **Structured API Query & JSON Validation**:
        ส่ง Request ไปยังเซิร์ฟเวอร์ด้วยหัวข้อความจำเพาะ โดยระบุอุณหภูมิโมเดลเป็น `temperature: 0` เพื่อผลลัพธ์ที่แน่นอนและเสถียรที่สุด และบังคับควบคุมโครงสร้างผลลัพธ์ด้วยการส่ง **JSON Schema** เข้าไปในระบบ ได้แก่:
        - `text_analysis`: ประเมินความเกี่ยวข้อง (`informativeness` เป็น `informative` / `not_informative`) และจำแนกกลุ่มความช่วยเหลือมนุษยธรรมตามเกณฑ์ (`category` 8 คลาส)
        - `image_analysis`: ประเมินฝั่งรูปภาพและระบุระดับความเสียหายของโครงสร้างที่มองเห็นได้ (`damage_severity` เช่น `mild_damage`, `severe_damage`)
     4. **Parallel Pipeline & Batch Processing**:
        โค้ดใช้คลาส `ThreadPoolExecutor` ของ Python ในการแบ่งสัดส่วนชุดข้อมูลและยิงคำขอไปยังโมเดลแบบคู่ขนาน (Multithreading) เพื่อประหยัดเวลาในการประมวลผลชุดข้อมูลขนาดใหญ่ พร้อมฟังก์ชัน Backup ผลลัพธ์เป็นไฟล์ Zip อัตโนมัติทุก 15 นาที
     5. **Evaluation Metrics**:
        เปรียบเทียบผลลัพธ์ที่ทำนายได้จากโมเดลกับข้อมูลเฉลย (Ground Truth) โดยคำนวณหาค่า F1-Score ในระดับต่าง ๆ ด้วย `sklearn.metrics.f1_score` และวาดภาพ Confusion Matrices เพื่อดูความสอดคล้องความแม่นยำในการแยกแยะหมวดหมู่ต่าง ๆ

   - **การปรับปรุงและประยุกต์ใช้สำหรับระบบของเรา (Text-Only Adaptation)**:
     เนื่องจากระบบแจ้งเตือนภัยของเรา**ไม่มีการรับหรือประมวลผลรูปภาพ (Text-Only)** เราจะดัดแปลงแนวคิดจากเปเปอร์นี้มาใช้งานเฉพาะในส่วนของข้อความดังนี้:
     1. **การยกเว้นขั้นตอนทางรูปภาพ**: ตัดส่วน *Image Preprocessing* และการประเมิน *image_analysis* (เช่น `damage_severity` และหมวดหมู่รูปภาพ) ออกทั้งหมด เพื่อให้โมเดลโฟกัสเฉพาะการสกัดรายละเอียดตัวอักษร
     2. **การปรับใช้ระบบ API**: เราจะใช้ 3 โมเดล MoE (deepseek-v4-flash, typhoon-v2.5, Gemma 4) ประมวลผลข้อความภาษาไทยผ่าน Client SDK และคีย์ API ที่เกี่ยวข้องของแต่ละรุ่น ในลักษณะเดียวกับการทดลองอื่น ๆ
     3. **Structured Text API Query**: บังคับโครงสร้างผลลัพธ์ด้วยการทำ **Function Calling (Function Call)** เพื่อสกัดรายละเอียดข้อมูล (`data` ที่มีรายละเอียดบุคคล สิ่งของ และผู้ประสบภัย) โดยตั้งค่า `temperature: 0` เพื่อความเป็นระเบียบและเสถียรของคำตอบเช่นเดียวกัน
     4. **การประยุกต์ใช้ 2-Stage Pipeline (Informativeness & Level Labeling)**:
        - *ขั้นตอนที่ 1*: คัดกรองข้อความขยะ/ทั่วไปที่ไม่เกี่ยวข้องกับภัยพิบัติก่อน (Informativeness Detection)
        - *ขั้นตอนที่ 2*: นำข้อความที่ผ่านตัวกรองไปสกัดข้อมูล NER

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จากชุดข้อมูลสังเคราะห์ทั้งหมด 2,000 รายการของ Experiment 04 จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:

```text
e:/nlp-for-disaster/exp4/results/
├── deepseek-v4-flash_results.csv        <- บันทึกผลการวิเคราะห์ NER
├── typhoon-v2.5_results.csv             <- บันทึกผลการวิเคราะห์ NER
├── gemma-4_results.csv                  <- บันทึกผลการวิเคราะห์ NER
└── model_comparison_metrics.csv         <- สรุปเปรียบเทียบผลลัพธ์ความถูกต้อง
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema)
ไฟล์ผลลัพธ์แยกตามรุ่นโมเดล (`deepseek-v4-flash_results.csv`, `typhoon-v2.5_results.csv`, `gemma-4_results.csv`) จะจัดเก็บประวัติการทำนายข้อมูล NER ที่สกัดได้จากข้อความแปลภาษาไทย โดยมีโครงสร้างคอลัมน์ดังนี้:

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `tweet_id` | ไอดีข้อความทวีต (ตรงตามชุดข้อมูลต้นฉบับ) | `8.29177E+17` |
| `translated_thai` | ข้อความโซเชียลมีเดียภาษาไทยที่แปลแล้วที่ส่งให้โมเดลวิเคราะห์ | *“ด่วนที่สุด! น้ำล้นตลิ่งทะลักท่วมตัวเมืองเชียงราย...”* |
| `true_text_info` | เฉลยจริง: ความเกี่ยวข้องภัยพิบัติ (Ground Truth) | `informative` / `not_informative` |
| `true_text_human` | เฉลยจริง: หมวดหมู่ช่วยเหลือทางมนุษยธรรม (Ground Truth) | `infrastructure_and_utility_damage` / `not_humanitarian` |
| `predicted_message_detail` | รายละเอียดเหตุการณ์ที่สกัดได้ (ฟิลด์ `message_more_detail`) | *“น้ำล้นตลิ่งทะลักท่วมตัวเมือง...”* |
| `predicted_victims_dead` | จำนวนผู้เสียชีวิตที่สกัดได้ (ฟิลด์ `victims.dead`) | `0` |
| `predicted_victims_critical` | จำนวนผู้ประสบภัยขั้นวิกฤตที่สกัดได้ (ฟิลด์ `victims.critical`) | `1` |
| `predicted_victims_urgent` | จำนวนผู้ประสบภัยต้องการช่วยเหลือด่วน (ฟิลด์ `victims.urgent`) | `0` |
| `predicted_victims_safe` | จำนวนผู้ประสบภัยที่ปลอดภัย (ฟิลด์ `victims.safe`) | `2` |
| `predicted_victims_child` | จำนวนเด็กที่สกัดได้ (ฟิลด์ `victims.child`) | `1` |
| `predicted_victims_bedridden` | จำนวนผู้ป่วยติดเตียงที่สกัดได้ (ฟิลด์ `victims.bedridden`) | `0` |
| `predicted_items_firstaid` | ความต้องการชุดปฐมพยาบาล (ฟิลด์ `items.firstAid`) | `1` |
| `predicted_items_food` | ความต้องการอาหารและน้ำ (ฟิลด์ `items.food`) | `1` |
| `predicted_items_energy` | ความต้องการแหล่งพลังงาน/ไฟสำรอง (ฟิลด์ `items.energy`) | `0` |
| `predicted_location` | ชื่อสถานที่ที่ระบุในข้อความที่สกัดได้ (ฟิลด์ `coordinates.name`) | *“ตัวเมืองเชียงราย”* |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษต้นฉบับ | `42` |
| `translated_thai_char_count` | จำนวนตัวอักษรของข้อความแปลภาษาไทย `translated_thai` | `65` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผลสะสมในระบบเอเจนต์ | `310` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผลสะสมในระบบเอเจนต์ | `185` |
