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

## 4. การออกแบบคำสั่ง (Two-Layer Prompt Design)

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะใช้เป็นภาษาอังกฤษถอดแบบมาจากงานวิจัยดั้งเดิม (ref/disaster_classification.ipynb) 100% เพื่อควบคุมตัวแปรในการทดลอง:

### 4.1 คำสั่งระบบ (System Instruction)
```markdown
You are an expert humanitarian disaster analyst with extensive experience in classifying disaster-related content. Your task is to accurately classify tweets and images based on objective evidence rather than emotional responses.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Tweet: "{tweet_text}"

CLASSIFICATION CRITERIA:

STEP 1: Analyze the TWEET TEXT first:
--------------------------------------
1. Tweet informativeness - determine if the tweet contains SPECIFIC information:
   - informative: Contains SPECIFIC disaster impact/response evidence, facts, or details
   - not_informative: Generic statements, emotions only, unrelated content, no specific details

2. Tweet category - identify the DOMINANT content (choose only ONE):
   - affected_individuals: Mentions displaced people, survivors, emotional responses (NOT injured/dead)
   - infrastructure_and_utility_damage: References damaged buildings, roads, bridges, utilities
   - injured_or_dead_people: Reports injuries, deaths, or specific casualty numbers
   - missing_or_found_people: Mentions people who are missing, found, or rescued by name or count
   - not_humanitarian: Irrelevant content, ads, jokes, political messages, misinformation
   - other_relevant_information: Weather data, satellite images, locations without people/damage
   - rescue_volunteering_or_donation_effort: Mentions donations, rescue missions, aid, volunteers
   - vehicle_damage: References damaged cars, trucks, ambulances, buses

Return classification by calling the specified function.
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
