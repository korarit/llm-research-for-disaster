# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Disaster Alert Labeling Experiment - 04)

เอกสารนี้รวบรวมข้อสรุปแนวทาง โครงสร้างการสกัดชื่อเฉพาะ (Named Entity Recognition - NER) และแนวทางการเขียน Prompt สำหรับการทดสอบร่วมกับ Large Language Model (LLM) ในเบื้องต้น สำหรับการทดลอง **Experiment 04**

---

## 1. โครงสร้างการสกัดชื่อจำเพาะ (Named Entity Recognition - NER)

ในการสกัด Entity จากข้อความภัยพิบัติภาษาไทย จะมุ่งเน้นไปที่ข้อมูลหลัก 4 ประเภท:
1. **Disaster Type (ประเภทภัยพิบัติ)**: เช่น น้ำท่วม, น้ำป่าไหลหลาก, แผ่นดินไหว, ฝุ่น PM2.5, ไฟไหม้
2. **Location (สถานที่เกิดเหตุ)**: ระบุขอบเขตพื้นที่ที่ได้รับผลกระทบ เช่น ถนน, ตำบล, อำเภอ, จังหวัด
3. **Datetime (วันและเวลา)**: ช่วงเวลาที่เกิดเหตุ หรือช่วงเวลาที่การคาดการณ์จะมีผลบังคับใช้
4. **Impact (ผลกระทบ)**: ความเสียหายที่ระบุ เช่น ถนนขาด, เสาไฟล้ม, จำนวนผู้บาดเจ็บ, ระดับน้ำความสูง

---

## 2. การกำหนดค่าโมเดลและสถาปัตยกรรมระบบ (Model & Pipeline Architecture)

ระบบการทดลองจะประยุกต์ใช้โมดูลและไลบรารีในลักษณะเดิมเพื่อทำการทดสอบโมเดล MoE ทั้ง 3 รุ่นร่วมกับชุดข้อมูลภัยพิบัติภาษาไทย:

### 2.1 การเชื่อมต่อโมเดลผ่าน API
การเชื่อมต่อโมเดลทุกตัวจะใช้การเรียก API ภายนอก (External API Call) ทั้งหมดผ่าน OpenAI-compatible client SDK โดยอ้างอิง Endpoint และคีย์ผู้ให้บริการดังนี้:

1. **Gemma 4** (เข้าถึงผ่าน OpenRouter API):
   - **โมเดลคีย์:** `google/gemma-4-26b-a4b-it`
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("OPENROUTER_API_KEY")`
   - **Endpoint:** `https://openrouter.ai/api/v1`

2. **deepseek-v4-flash** (เข้าถึงผ่าน OpenRouter API):
   - **โมเดลคีย์:** `deepseek/deepseek-v4-flash`
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("OPENROUTER_API_KEY")`
   - **Endpoint:** `https://openrouter.ai/api/v1`

3. **typhoon-v2.5** (เข้าถึงผ่าน Typhoon API):
   - **โมเดลคีย์:** `typhoon-v2.5-30b-a3b-instruct`
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("TYPHOON_API_KEY")`
   - **Endpoint:** `https://api.opn.ai/v1`

### 2.2 รูปแบบฟังก์ชันเรียกใช้งาน (Function Calling Schema)
ในการประมวลผลข้อมูลผ่าน API จะใช้คุณลักษณะการเรียกฟังก์ชัน (Function Calling) โดยส่งฟังก์ชันที่มีอาร์กิวเมนต์ย่อยแบบแบนเรียบหลายตัว (Flat parameters) เข้าไปยังโมเดลโดยตรง เพื่อให้โมเดลกรอกข้อมูลลงในฟิลด์ต่าง ๆ โดยตรง (ไม่ใช่การส่ง Raw JSON String คืนกลับมาในพารามิเตอร์เดียว) โดยกำหนดโครงสร้างข้อมูลตามคลาส Pydantic ดังนี้:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ContactDetail(BaseModel):
    name: Optional[str] = Field(description="Full name or first name if found, otherwise null")
    nickname: Optional[str] = Field(description="Nickname if found, otherwise null")
    phone: Optional[str] = Field(description="Phone number found in the tweet, otherwise null")

class VictimsCount(BaseModel):
    dead: int = Field(default=0, description="Number of dead people explicitly reported")
    critical: int = Field(default=0, description="Number of people trapped, missing, in severe danger or severely injured")
    urgent: int = Field(default=0, description="Number of injured or sick people needing prompt assistance")
    safe: int = Field(default=0, description="Number of people reported safe/evacuated")
    child: int = Field(default=0, description="Number of children affected")
    infant: int = Field(default=0, description="Number of infants affected")

class ItemsCount(BaseModel):
    firstAid: int = Field(default=0, description="Quantity/Need of first-aid kits or medicine (1 if needed but quantity not specified)")
    food: int = Field(default=0, description="Quantity/Need of food/drinking water (1 if needed but quantity not specified)")
    energy: int = Field(default=0, description="Quantity/Need of flashlights, powerbanks, candles, or backup power (1 if needed but quantity not specified)")

class CoordinatesDetail(BaseModel):
    name: Optional[str] = Field(description="Specific location name, landmark, road, or sub-district name mentioned in the tweet")
    google_map_url: Optional[str] = Field(default=None)
    lat: float = Field(default=0.0)
    lng: float = Field(default=0.0)

class NERResult(BaseModel):
    message_more_detail: str = Field(description="Brief summary of the disaster incident details in Thai")
    contact_victim: List[ContactDetail]
    contact_reporter: List[ContactDetail]
    victims: VictimsCount
    items: ItemsCount
    coordinates: CoordinatesDetail
```

### 2.3 หมายเหตุสำคัญเกี่ยวกับการทดลอง
1. **สกัดข้อมูลจากข้อความ**: ข้อมูลบุคคล (`contact_victim`, `contact_reporter`), ข้อมูลผู้ประสบภัย (`victims`), และสิ่งของที่ต้องการ (`items`) จะถูกดึงและสรุปมาจากรายละเอียดในข้อความแจ้งเตือนดิบ
2. **อย่าพึ่งเขียนโค้ดสำหรับ coordinates**: ในส่วนของ `coordinates` (`lat`, `lng`, `google_map_url`) จะยังไม่เขียนโค้ดให้โมเดลทำภูมิสารสนเทศหรือระบุพิกัดละติจูด/ลองจิจูดในเวลานี้ (ให้โมเดลคืนค่าโครงสร้างเปล่าหรือใช้ค่าเริ่มต้นไปก่อน เช่น `"lat": 0.0, "lng": 0.0`)

---

## 3. การออกแบบคำสั่ง (Prompt Design Template) - นำ Prompt Concept จาก 01TH มาปรับใช้

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะอ้างอิงตาม Prompt Concept ของ **Experiment 01TH** โดยเขียนเป็นภาษาอังกฤษเพื่อประสิทธิภาพสูงสุดของโมเดล MoE และความคุ้มค่าด้าน Token (Token Cost Efficiency) แต่ทำการปรับปรุง Signal Words และ Edge Cases ให้เป็นคำภาษาไทยและตัวอย่างบริบทภาษาไทย เพื่อให้โมเดลสามารถวิเคราะห์และสกัดข้อมูลจากข้อความภาษาไทยได้อย่างถูกต้องและแม่นยำ:

### 3.1 คำสั่งระบบ (System Instruction)
```markdown
You are a disaster response information analyst. Your task is to analyze social media posts (tweets) or alerts about disasters in Thailand and extract key named entities.
```

### 3.2 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Tweet: "{text}"

Analyze the tweet and extract information according to the definitions and rules below.

INFORMATION EXTRACTION (NER) INSTRUCTIONS:
- message_more_detail: Briefly summarize additional situation details in Thai.
- contact_victim: List of victims mentioned in the text. For each victim, extract "name", "nickname", "phone" (if not mentioned, set to null or empty list).
- contact_reporter: List of reporters/informants. For each, extract "name", "nickname", "phone" (if not mentioned, set to null or empty list).
- victims: Count of victims by category based on details in the text.
  - dead: number of dead people reported
  - critical: number of victims in critical condition (e.g., trapped, swept away, severely injured)
  - urgent: number of victims needing urgent help (e.g., injured but stable, lacking supplies)
  - safe: number of survivors confirmed safe
  - child: number of children affected
  - infant: number of infants affected
  - If not specified, set counts to 0.
- items: Count/need of relief items. Set to number needed, or 1 if needed but count not specified, and 0 if not needed:
  - firstAid: first aid kits, medicine (ยารักษาโรค, ยา, ชุดปฐมพยาบาล)
  - food: food, drinking water (อาหาร, น้ำดื่ม, ข้าวกล่อง, ของกิน)
  - energy: power sources, flashlights, powerbanks, backup generators (ไฟสำรอง, แบตสำรอง, พาวเวอร์แบงค์, ไฟฉาย, เทียน, เครื่องปั่นไฟ)
- coordinates: Set "name" to the location name mentioned in the text. Set "google_map_url" to null, and "lat" to 0.0, "lng" to 0.0 (do not try to resolve coordinates).

Call the function 'extract_information' with the extracted details.
```

---

## 4. แหล่งอ้างอิงและที่มา (References & Sources)

1. **มาตรฐานการเตือนภัยสากล (CAP Standard)**
   - **Common Alerting Protocol (CAP) Standard (ITU Recommendation X.1303)**: กรอบมาตรฐานสากลสำหรับการจัดส่งโครงสร้างข้อมูลเตือนภัยและระดับภัยพิบัติ (Severity, Urgency, Certainty)
   - แหล่งข้อมูล: [ITU-T X.1303](https://www.itu.int/rec/T-REC-X.1303)

2. **เกณฑ์การตอบสนองภัยพิบัติของ FEMA**
   - **FEMA Incident Complexity & Activation Levels**: แนวทางการแบ่งระดับปฏิบัติการฉุกเฉิน (Level I, II, III) ตามระดับผลกระทบและความพร้อมด้านทรัพยากร
   - แหล่งข้อมูล: [FEMA Emergency Operations Center (EOC) Activation Levels](https://www.fema.gov/)

3. **งานวิจัยที่เกี่ยวข้องและแนวคิดหลัก**
   - **ชื่องานวิจัย**: [Zero-Shot Social Media Crisis Classification: A Training-Free Multimodal Approach](https://digibug.ugr.es/bitstream/handle/10481/111587/applsci-16-02192.pdf?sequence=1&isAllowed=y) (MDPI Applied Sciences, 2026)
   - **ซอร์สโค้ดอ้างอิงของงานวิจัย (Local Workspace)**: [disaster_classification.ipynb](file:///e:/nlp-for-disaster/ref/disaster_classification.ipynb)
   - **สกัดขั้นตอนการทำงานของซอร์สโค้ดต้นแบบ (How It Works)**:
     โค้ดเตรียมนิเวศการรันและโมเดลสำหรับประมวลผลข้อมูลด้วยขั้นตอนดังนี้:
     1. **Local LLM Server Hosting**:
        ดาวน์โหลดโมเดลขนาดใหญ่ `Mistral-Small-3.1-24B-Instruct-2503` (ฟอร์慢 GGUF ระดับ Q6_K_L) พร้อมโมดูลสแกนภาพ (Pixtral Vision mmproj) และรันเซิร์ฟเวอร์แบบ Local โดยใช้ `llama-server` ผ่านคอมมานด์ไลน์เพื่อสร้าง API จำลองสากลที่เข้ากันได้กับ OpenAI API (`/chat/completions`) บนพอร์ต `8000` โดยใช้ค่า Parameter `-c 24576` (Context Window) และการทำงานขนานแบบ `--parallel 16`
     2. **Image Preprocessing**:
        โค้ดจัดการรูปภาพจากชุดข้อมูลก่อนส่งให้โมเดลประมวลผล โดยแปลงสีรูปจาก RGBA/อื่น ๆ ไปเป็น RGB, ย่อขนาดภาพให้เท่ากันที่ `512x512 px` (JPEG, Quality 85%) แล้วเปลี่ยนไฟล์รูปเป็น Base64 String เพื่อจัดชุดคำสั่งแบบ Multimodal (Text + Image) ส่งตรงผ่าน API
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
| `predicted_victims_infant` | จำนวนเด็กทารกที่สกัดได้ (ฟิลด์ `victims.infant`) | `0` |
| `predicted_items_firstaid` | ความต้องการชุดปฐมพยาบาล (ฟิลด์ `items.firstAid`) | `1` |
| `predicted_items_food` | ความต้องการอาหารและน้ำ (ฟิลด์ `items.food`) | `1` |
| `predicted_items_energy` | ความต้องการแหล่งพลังงาน/ไฟสำรอง (ฟิลด์ `items.energy`) | `0` |
| `predicted_location` | ชื่อสถานที่ที่ระบุในข้อความที่สกัดได้ (ฟิลด์ `coordinates.name`) | *“ตัวเมืองเชียงราย”* |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษต้นฉบับ | `42` |
| `translated_thai_char_count` | จำนวนตัวอักษรของข้อความแปลภาษาไทย `translated_thai` | `65` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผลสะสมในระบบเอเจนต์ | `310` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผลสะสมในระบบเอเจนต์ | `185` |
