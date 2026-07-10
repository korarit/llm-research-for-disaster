# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติภาษาไทย (Disaster Alert Labeling Experiment - 01TH)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 01TH** ซึ่งเป็นการคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติ **ภาษาไทย** ด้วยวิธี **เลเยอร์เดียวขั้นตอนเดียว (Single-Layer / Flat Classification)** 

การทดลองนี้เป็นการนำชุดข้อมูลที่แปลไทยและแปลงบริบทเป็นประเทศไทยสำเร็จแล้วจำนวน 500 แถวจากไฟล์ [CrisisMMD_Thai_500.csv](file:///e:/nlp-for-disaster/data/CrisisMMD_Thai_500.csv) (ตามแผนงาน [plan_CrisisMMD_to_thai.md](file:///e:/nlp-for-disaster/plan_CrisisMMD_to_thai.md)) มาคัดแยกเพื่อประเมินความสามารถทางภาษาไทยของโมเดล MoE แต่ละรุ่น

---

## 1. วัตถุประสงค์ (Objectives)
- ประเมินขีดความสามารถการทำ Zero-shot classification ข้อความโซเชียลมีเดียภาษาไทยของกลุ่มโมเดล MoE ทั้ง 3 รุ่น (`deepseek-v4-flash`, `typhoon-v2.5`, `Gemma 4`)
- ศึกษาประสิทธิภาพและทดสอบระบบการคัดแยกแบบเลเยอร์เดียว (Single-Layer) ในภาษาไทย เพื่อวัดผลเป็นฐานข้อมูลหลัก (Baseline) ก่อนเริ่มทำการแยกสองเลเยอร์ในการทดลองถัดไป (02TH)

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล (Dataset & Preparation)
- **แหล่งข้อมูลเข้า (Input Source):** ไฟล์ข้อมูลแปลไทย `e:/nlp-for-disaster/data/CrisisMMD_Thai_500.csv`
- **โครงสร้างข้อมูลเข้าหลัก (Input Columns):**
  - `tweet_id`: ไอดีข้อความ
  - `translated_thai`: ข้อความโซเชียลมีเดียภาษาไทยที่ผ่านการแปลและแปลงสถานที่ในไทยอย่างเป็นธรรมชาติ
- **เป้าหมายคำตอบเฉลย (Ground Truth Columns):**
  - `true_text_info`: ระบุว่าข้อความเกี่ยวข้องหรือไม่ (`informative` / `not_informative`)
  - `true_text_human`: ระบุหมวดหมู่การช่วยเหลือทางมนุษยธรรมดั้งเดิม

---

## 3. การกำหนดค่าโมเดลและโครงสร้าง Function Calling (Single-Layer TH)

ระบบจะใช้โครงสร้าง Pydantic Schema แบบฟิลด์เดียวมาทำ **Function Calling (Function Call)** เพื่อความเสถียรของ Output ในภาษาไทย:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SingleLayerClassificationResultTH(BaseModel):
    category: Literal[
        "not_informative",
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "injured_or_dead_people",
        "missing_or_found_people",
        "other_relevant_information",
        "rescue_volunteering_or_donation_effort",
        "vehicle_damage"
    ] = Field(description="Choose the most specific category represented in the tweet")
```

---

## 4. การออกแบบคำสั่ง (Single-Layer Prompt Design) - นำ Prompt Concept จาก 1E (promptV3) มาปรับใช้

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะอ้างอิงตาม Prompt Concept ของ **Experiment 1E (promptV3)** โดยเขียนเป็นภาษาอังกฤษเพื่อประสิทธิภาพสูงสุดของโมเดล MoE และความคุ้มค่าด้าน Token (Token Cost Efficiency) แต่ทำการปรับปรุง Signal Words และ Edge Cases ให้เป็นคำภาษาไทยและตัวอย่างบริบทภาษาไทย เพื่อให้โมเดลสามารถวิเคราะห์ข้อความภาษาไทยได้อย่างถูกต้องและแม่นยำ:

### 4.1 คำสั่งระบบ (System Instruction)
```markdown
You are a humanitarian disaster information analyst. Your task is to classify social media posts (tweets) collected during disaster events into exactly one category for emergency response analysis.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Tweet: "{tweet_text}"

Classify the tweet into exactly ONE of the following categories. Choose the category that best represents the primary subject of the tweet:

CATEGORY DEFINITIONS:
1. not_informative: Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.
2. injured_or_dead_people: Reports of casualties, deaths, fatalities, injuries, or hospitalized people. (Thai signal words: เสียชีวิต, ตาย, เสียชีวิตแล้ว, ผู้เสียชีวิต, พบร่าง, พบศพ, ยอดเสียชีวิต, บาดเจ็บ, ได้รับบาดเจ็บ, เจ็บ, เจ็บสาหัส, บาดเจ็บสาหัส, ส่งโรงพยาบาล, ส่งรพ., รักษาตัวที่โรงพยาบาล, กู้ชีพพบร่าง)
3. missing_or_found_people: Reports of specific individuals or groups who are missing, active searches, or confirmed rescues. (Thai signal words: สูญหาย, หายตัว, หาย, สูญหายไป, ตามหา, ค้นหา, ค้นหาผู้สูญหาย, พบตัวแล้ว, เจอแล้ว, พบตัว, ช่วยชีวิตได้แล้ว, ช่วยเหลือได้แล้ว, รอดชีวิต, ปลอดภัยแล้ว, ติดต่อไม่ได้, ยังไม่พบตัว, ขาดการติดต่อ)
4. affected_individuals: Evacuees, displaced people, survivors, homeless, stranded, or those taking shelter (WITHOUT reported deaths or injuries). (Thai signal words: อพยพ, ถูกอพยพ, หนีน้ำ, พลัดถิ่น, ไร้ที่อยู่อาศัย, ไร้บ้าน, ศูนย์อพยพ, สถานที่พักพิง, จุดพักพิง, ติดอยู่, ติดค้าง, ออกไม่ได้, ผู้รอดชีวิต, ผู้ประสบภัย, ชาวบ้านเดือดร้อน, บ้านน้ำท่วม)
5. infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power grids, water lines, or utility outages. (Thai signal words: ถล่ม, ทรุดตัว, พังทลาย, ยุบตัว, พังเสียหาย, ได้รับความเสียหาย, เสียหาย, พัง, ไฟดับ, น้ำประปาไม่ไหล, ไม่มีไฟฟ้า, สัญญาณขาดหาย, น้ำท่วม, ถนนถูกน้ำท่วม, ท่วมถนน, น้ำท่วมขัง, ถนนขาด, ถนนพัง, สะพานขาด, สะพานพัง, เส้นทางชำรุด, เสาไฟล้ม, อาคารถล่ม)
6. vehicle_damage: Damage to cars, trucks, boats, trains, or planes as the primary subject. (Thai signal words: รถจมน้ำ, รถยนต์จมน้ำ, รถพัง, รถยนต์เสียหาย, รถได้รับความเสียหาย, เรือล่ม, เรือพัง, เรืออับปาง, เรือจม, รถไหลไปกับน้ำ, รถโดนพัดไป, รถคว่ำ, รถบรรทุกคว่ำ)
7. rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, rescue operations, or emergency helpline sharing. (Thai signal words: บริจาค, เปิดรับบริจาค, เงินบริจาค, สมทบทุน, ระดมทุน, ร่วมบริจาค, จิตอาสา, อาสาสมัคร, อาสา, ถุงยังชีพ, ข้าวกล่อง, แจกของ, ความช่วยเหลือ, สิ่งของช่วยเหลือ, แจกจ่ายสิ่งของ, หน่วยกู้ภัย, ทีมกู้ภัย, กู้ภัย, กู้ชีพ, ลงพื้นที่ช่วยเหลือ)
8. other_relevant_information: General news, weather forecasts, warning alerts, magnitude reports, or opinions about the disaster that do not report specific human or physical impact. (Thai signal words: เตือนภัย, ประกาศเตือน, เฝ้าระวัง, แจ้งเตือน, ประกาศจากราชการ, พยากรณ์อากาศ, คาดการณ์, ดินฟ้าอากาศ, ความรุนแรง, ริกเตอร์, ขนาดความแรง, ระดับน้ำ, ปริมาณน้ำฝน, ข่าวภัยพิบัติ, รายงานสถานการณ์, อัพเดทสถานการณ์, อัพเดทน้ำท่วม, ภาพดาวเทียม, เส้นทางพายุ, พายุเข้า)

CRITICAL DECISION HIERARCHY (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_informative (Only for completely irrelevant or generic sentiment posts with no disaster details)

EDGE-CASE RESOLUTION RULES:
- "ส่งกำลังใจให้ผู้ประสบภัยน้ำท่วมเชียงราย #น้ำท่วมเชียงราย" -> Classify as 'other_relevant_information' (contains specific disaster keyword).
- "ขอส่งกำลังใจให้ทุกคนปลอดภัย" -> Classify as 'not_informative' (no specific disaster reference or details).
- "ศูนย์อพยพวัดศรีทรายมูลกำลังแจกข้าวกล่องและน้ำดื่ม" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "สะพานพัง รถสัญจรผ่านไม่ได้ที่แม่สาย" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

Return classification by calling the specified function.
```

---

## 5. แผนการวัดผลและการจัดเก็บข้อมูล
- **การแปลงข้อมูลกลับ (Mapping):** ทำการแปลงผลลัพธ์ของคลาส `not_informative` ไปเป็น `not_informative` (Informativeness) และ `not_humanitarian` (Category) เพื่อนำไปคำนวณ F1-Score ภาพรวมของภาษาไทย
- **โครงสร้างไฟล์ผลลัพธ์:**
  ```text
  e:/nlp-for-disaster/exp1_TH/results/
  ├── deepseek-v4-flash_results_th.csv
  ├── typhoon-v2.5_results_th.csv
  ├── gemma-4_results_th.csv
  ├── th_model_comparison_metrics.csv
  ├── th_model_comparison_chart.png
  └── confusion_matrices/
  ```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema - TH)
ไฟล์ผลลัพธ์แยกตามรุ่นโมเดลสำหรับภาษาไทย (`deepseek-v4-flash_results_th.csv`, `typhoon-v2.5_results_th.csv`, `gemma-4_results_th.csv`) จะจัดเก็บคำทำนาย (Label) ของ AI คู่ขนานไปกับข้อความแปลภาษาไทยและเฉลยจริง โดยประกอบไปด้วยคอลัมน์ดังนี้:

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `tweet_id` | ไอดีข้อความทวีต (ตรงตามชุดข้อมูลต้นฉบับ) | `8.29177E+17` |
| `translated_thai` | ข้อความโซเชียลมีเดียภาษาไทยที่แปลแล้วที่ส่งให้โมเดลวิเคราะห์ | *“ขอแรงใจให้เชียงรายด้วยครับ ตอนนี้บ้านผมน้ำท่วมสูงมาก...”* |
| `true_text_info` | เฉลยจริง: ความเกี่ยวข้องภัยพิบัติ (Ground Truth) | `informative` / `not_informative` |
| `true_text_human` | เฉลยจริง: หมวดหมู่ช่วยเหลือทางมนุษยธรรม (Ground Truth) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `predicted_category` | ผลการคัดแยกโดยตรงที่ AI ตอบกลับมาตาม Function Calling (Single-Layer TH) | `rescue_volunteering_or_donation_effort` / `not_informative` |
| `mapped_predicted_info` | ผลความเกี่ยวข้อง (Informativeness) ที่ได้หลังการแปลงผล (Mapping) | `informative` / `not_informative` |
| `mapped_predicted_category` | ผลหมวดหมู่ช่วยเหลือ (Category) ที่ได้หลังการแปลงผล (Mapping) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษต้นฉบับ | `42` |
| `translated_thai_char_count` | จำนวนตัวอักษรของข้อความแปลภาษาไทย `translated_thai` | `65` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผล | `185` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผล | `15` |
