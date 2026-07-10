# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติภาษาไทย (Disaster Alert Labeling Experiment - 02TH)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 02TH** ซึ่งเป็นการคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติ **ภาษาไทย** ด้วยวิธี **สองเลเยอร์ร่วมกัน (Two-Layer Joint Classification)** 

การทดลองนี้เป็นการนำข้อมูลข้อความแปลภาษาไทยจำนวน 500 แถวที่เป็น **ชุดข้อมูลชุดเดียวกันกับที่รันใน Experiment 01TH ทุกประการ** เพื่อเปรียบเทียบเชิงประสิทธิภาพระหว่างการคัดแยกแบบเลเยอร์เดียวและแบบสองเลเยอร์

---

## 1. วัตถุประสงค์ (Objectives)
- ประเมินขีดความสามารถการทำ Zero-shot classification แบบสองเลเยอร์แยกกันในก้อนผลลัพธ์เดียว (Two-Layer Joint Prediction) ในภาษาไทย ของกลุ่มโมเดล MoE ทั้ง 3 รุ่น (`deepseek-v4-flash`, `typhoon-v2.5`, `Gemma 4`)
- เปรียบเทียบค่าความแม่นยำ F1-Score ระหว่างสถาปัตยกรรมแบบ 1-Layer (จาก Exp 01TH) และ 2-Layer (Exp 02TH) เพื่อวิเคราะห์สถาปัตยกรรมคำสั่ง (Prompt) ที่ได้ประสิทธิภาพสูงสุดในการเข้าใจบริบทภัยพิบัติภาษาไทย

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล (Dataset & Preparation)
- **แหล่งข้อมูลเข้า (Input Source):** ดึงชุดข้อความและเฉลยความจริงโดยตรงจาก **ไฟล์ผลลัพธ์ข้อมูลทดสอบ 500 แถวที่บันทึกสำเร็จมาจาก Experiment 01TH**
- **ความโปร่งใสและถูกต้อง:** **จะไม่มีการสุ่มชุดข้อมูลแปลไทยใหม่ในการทดลองนี้** เพื่อควบคุมปัจจัยแวดล้อมให้ทดสอบบนข้อความทวีตเดียวกัน 100% (Apple-to-Apple Comparison)

---

## 3. การกำหนดค่าโมเดลและโครงสร้าง Function Calling (Two-Layer TH)

ระบบจะใช้ Pydantic Schema แบบ 2 เลเยอร์มาทำ **Function Calling (Function Call)** โดยกำหนดเงื่อนไขการประเมินให้ระบุทั้งระดับความเกี่ยวข้อง (Informativeness) และหมวดหมู่การช่วยเหลือ (Category) ร่วมกัน:

```python
from pydantic import BaseModel, Field
from typing import Literal

class TwoLayerClassificationResultTH(BaseModel):
    informativeness: Literal["informative", "not_informative"] = Field(
        description="determine if the tweet contains SPECIFIC disaster impact/response evidence, facts, or details"
    )
    category: Literal[
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "injured_or_dead_people",
        "missing_or_found_people",
        "not_humanitarian",
        "other_relevant_information",
        "rescue_volunteering_or_donation_effort",
        "vehicle_damage"
    ] = Field(description="identify the DOMINANT content (choose only ONE)")
```

---

## 4. การออกแบบคำสั่ง (Two-Layer Prompt Design) - นำ Prompt Concept จาก 02E มาปรับใช้

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะอ้างอิงตาม Prompt Concept ของ **Experiment 02E** โดยเขียนเป็นภาษาอังกฤษเพื่อประสิทธิภาพการประมวลผลสูงสุดและประหยัด Token แต่ได้ทำการปรับปรุง Signal Words และ Edge Cases ให้เหมาะสมกับภาษาไทยเพื่อให้โมเดลสามารถทำงานร่วมกัน 2 เลเยอร์ได้อย่างสมบูรณ์:

### 4.1 คำสั่งระบบ (System Instruction)
```markdown
You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or generic personal banter/sentiment (prayers/wishes) that does not refer to the specific disaster at all.

LAYER 2 — HUMANITARIAN CATEGORY
Identify the category that best represents the primary subject of the tweet:
- injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Thai signal words: เสียชีวิต, ตาย, เสียชีวิตแล้ว, ผู้เสียชีวิต, พบร่าง, พบศพ, ยอดเสียชีวิต, บาดเจ็บ, ได้รับบาดเจ็บ, เจ็บ, เจ็บสาหัส, บาดเจ็บสาหัส, ส่งโรงพยาบาล, ส่งรพ., รักษาตัวที่โรงพยาบาล, กู้ชีพพบร่าง)
- missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Thai signal words: สูญหาย, หายตัว, หาย, สูญหายไป, ตามหา, ค้นหา, ค้นหาผู้สูญหาย, พบตัวแล้ว, เจอแล้ว, พบตัว, ช่วยชีวิตได้แล้ว, ช่วยเหลือได้แล้ว, รอดชีวิต, ปลอดภัยแล้ว, ติดต่อไม่ได้, ยังไม่พบตัว, ขาดการติดต่อ)
- affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Thai signal words: อพยพ, ถูกอพยพ, หนีน้ำ, พลัดถิ่น, ไร้ที่อยู่อาศัย, ไร้บ้าน, ศูนย์อพยพ, สถานที่พักพิง, จุดพักพิง, ติดอยู่, ติดค้าง, ออกไม่ได้, ผู้รอดชีวิต, ผู้ประสบภัย, ชาวบ้านเดือดร้อน, บ้านน้ำท่วม)
- infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Thai signal words: ถล่ม, ทรุดตัว, พังทลาย, ยุบตัว, พังเสียหาย, ได้รับความเสียหาย, เสียหาย, พัง, ไฟดับ, น้ำประปาไม่ไหล, ไม่มีไฟฟ้า, สัญญาณขาดหาย, น้ำท่วม, ถนนถูกน้ำท่วม, ท่วมถนน, น้ำท่วมขัง, ถนนขาด, ถนนพัง, สะพานขาด, สะพานพัง, เส้นทางชำรุด, เสาไฟล้ม, อาคารถล่ม)
- vehicle_damage: Damage to cars, trucks, boats, or planes. (Thai signal words: รถจมน้ำ, รถยนต์จมน้ำ, รถพัง, รถยนต์เสียหาย, รถได้รับความเสียหาย, เรือล่ม, เรือพัง, เรืออับปาง, เรือจม, รถไหลไปกับน้ำ, รถโดนพัดไป, รถคว่ำ, รถบรรทุกคว่ำ)
- rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Thai signal words: บริจาค, เปิดรับบริจาค, เงินบริจาค, สมทบทุน, ระดมทุน, ร่วมบริจาค, จิตอาสา, อาสาสมัคร, อาสา, ถุงยังชีพ, ข้าวกล่อง, แจกของ, ความช่วยเหลือ, สิ่งของช่วยเหลือ, แจกจ่ายสิ่งของ, หน่วยกู้ภัย, ทีมกู้ภัย, กู้ภัย, กู้ชีพ, ลงพื้นที่ช่วยเหลือ)
- other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Thai signal words: เตือนภัย, ประกาศเตือน, เฝ้าระวัง, แจ้งเตือน, ประกาศจากราชการ, พยากรณ์อากาศ, คาดการณ์, ดินฟ้าอากาศ, ความรุนแรง, ริกเตอร์, ขนาดความแรง, ระดับน้ำ, ปริมาณน้ำฝน, ข่าวภัยพิบัติ, รายงานสถานการณ์, อัพเดทสถานการณ์, อัพเดทน้ำท่วม, ภาพดาวเทียม, เส้นทางพายุ, พายุเข้า)
- not_humanitarian: Use this ONLY if the tweet is classified as 'not_informative' in Layer 1.

CRITICAL DECISION HIERARCHY FOR LAYER 2 (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_humanitarian (Only when Layer 1 is 'not_informative')

CONSISTENCY RULE:
- If Layer 1 is 'not_informative', Layer 2 must be 'not_humanitarian'.
- If Layer 1 is 'informative', Layer 2 must NOT be 'not_humanitarian' (choose one of the other 7 categories instead).

EDGE-CASE RESOLUTION RULES:
- "ส่งกำลังใจให้ผู้ประสบภัยน้ำท่วมเชียงราย #น้ำท่วมเชียงราย" -> Layer 1: 'informative', Layer 2: 'other_relevant_information' (contains specific disaster keyword).
- "ขอส่งกำลังใจให้ทุกคนปลอดภัย" -> Layer 1: 'not_informative', Layer 2: 'not_humanitarian' (no specific disaster reference).
- "ศูนย์อพยพวัดศรีทรายมูลกำลังแจกข้าวกล่องและน้ำดื่ม" -> Layer 1: 'informative', Layer 2: 'rescue_volunteering_or_donation_effort'.
- "สะพานขาด รถสัญจรไม่ได้ที่แม่สาย" -> Layer 1: 'informative', Layer 2: 'infrastructure_and_utility_damage'.

Return classification by calling the function 'classify_two_layer' with both values.
```

---

## 5. แผนการวัดผลและการจัดเก็บข้อมูล
- **การเปรียบเทียบข้ามสถาปัตยกรรม:** คำนวณ F1-Score สำหรับข้อความภาษาไทย และเขียนสคริปต์เปรียบเทียบผลลัพธ์ระหว่าง Exp 01TH และ Exp 02TH (`th_exp1_vs_exp2_comparison.csv`) เพื่อพิสูจน์ประสิทธิภาพในการประมวลผลข้อความภาษาไทยแบบ Flat vs Hierarchical
- **โครงสร้างไฟล์ผลลัพธ์:**
  ```text
  e:/nlp-for-disaster/exp2_TH/results/
  ├── deepseek-v4-flash_results_th.csv
  ├── typhoon-v2.5_results_th.csv
  ├── gemma-4_results_th.csv
  ├── th_model_comparison_metrics.csv
  ├── th_model_comparison_chart.png
  ├── th_exp1_vs_exp2_comparison.csv       <- เปรียบเทียบผลประสิทธิภาพระหว่าง 1-Layer และ 2-Layer ภาษาไทย
  └── confusion_matrices/
  ```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema - TH)
ไฟล์ผลลัพธ์แยกตามรุ่นโมเดลสำหรับภาษาไทย (`deepseek-v4-flash_results_th.csv`, `typhoon-v2.5_results_th.csv`, `gemma-4_results_th.csv`) สำหรับสถาปัตยกรรม Two-Layer Joint Classification จะจัดเก็บคำทำนายของ AI ทั้งสองเลเยอร์แยกกัน โดยมีโครงสร้างคอลัมน์ดังนี้:

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `tweet_id` | ไอดีข้อความทวีต (ตรงตามชุดข้อมูลต้นฉบับ) | `8.29177E+17` |
| `translated_thai` | ข้อความโซเชียลมีเดียภาษาไทยที่แปลแล้วที่ส่งให้โมเดลวิเคราะห์ | *“ขอแรงใจให้เชียงรายด้วยครับ ตอนนี้บ้านผมน้ำท่วมสูงมาก...”* |
| `true_text_info` | เฉลยจริง: ความเกี่ยวข้องภัยพิบัติ (Ground Truth) | `informative` / `not_informative` |
| `true_text_human` | เฉลยจริง: หมวดหมู่ช่วยเหลือทางมนุษยธรรม (Ground Truth) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `predicted_info` | คำทำนาย AI เลเยอร์ 1: ความเกี่ยวข้องภัยพิบัติ (ฟิลด์ `informativeness` จาก Function Call) | `informative` / `not_informative` |
| `predicted_category` | คำทำนาย AI เลเยอร์ 2: หมวดหมู่ช่วยเหลือ (ฟิลด์ `category` จาก Function Call) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษต้นฉบับ | `42` |
| `translated_thai_char_count` | จำนวนตัวอักษรของข้อความแปลภาษาไทย `translated_thai` | `65` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผล | `185` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผล | `25` |
