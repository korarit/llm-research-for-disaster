# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Disaster Alert Labeling Experiment - 02)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับการทดสอบ **Experiment 02** ซึ่งเป็นการปรับปรุงระบบคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **สองเลเยอร์ร่วมกัน (Two-Layer Joint Classification)** โดยอิงตามแนวทางการประเมินผลดั้งเดิมของงานวิจัย [Zero-Shot Social Media Crisis Classification: A Training-Free Multimodal Approach](https://digibug.ugr.es/bitstream/handle/10481/111587/applsci-16-02192.pdf?sequence=1&isAllowed=y) (ซอร์สโค้ดต้นฉบับ: [disaster_classification.ipynb](file:///e:/nlp-for-disaster/ref/disaster_classification.ipynb)) 

ในการทดลองที่ 2 นี้ จะนำข้อมูลผลการทดสอบของโมเดลกลุ่ม Mixture of Experts (MoE) 3 รุ่นเดิม จากชุดข้อมูลสุ่ม 500 รายการเดียวกัน มาคัดแยกและเปรียบเทียบในโครงสร้างแบบ 2 เลเยอร์ เพื่อศึกษาผลกระทบของการแยกทาสก์เปรียบเทียบกับแบบเลเยอร์เดียว (Experiment 01)
การทดลองนี้ประเมินเพิ่มเติมที่อุณหภูมิ (Temperature) **0.1, 0.2, 0.3** ควบคู่ไปกับค่าเริ่มต้น **0.0**

---

## 1. วัตถุประสงค์การทดลอง (Objectives)
- ประเมินประสิทธิภาพการทำ Zero-shot classification แบบสองเลเยอร์แยกกันในผลลัพธ์เดียว (Two-Layer Joint Prediction) ของกลุ่มโมเดล MoE ทั้ง 3 รุ่น (`deepseek/deepseek-v4-flash`, `typhoon-v2.5-30b-a3b-instruct`, `google/gemma-4-26b-a4b-it`)
- เปรียบเทียบ F1-Score และประสิทธิภาพการคัดแยกหมวดหมู่ย่อยระหว่างแบบขั้นตอนเดียว (จาก Experiment 01) และแบบสองขั้นตอนร่วมกัน (Experiment 02) เพื่อหาข้อสรุปแนวทางที่ดีที่สุดในแง่ของสถาปัตยกรรม Prompt
- วิเคราะห์ความแม่นยำในการระบุข้อความขยะแยกจากการระบุหมวดหมู่การบรรเทาทุกข์ในภาพรวม

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล (Dataset & Preparation)
ในการทดลองที่ 2 นี้ **จะไม่มีการสุ่มชุดข้อมูลใหม่** เพื่อให้เกิดความแม่นยำในการวัดผลเปรียบเทียบประสิทธิภาพแบบรายแถว (Apple-to-Apple Comparison):

- **แหล่งข้อมูลเข้า (Input Source):** สคริปต์จะดึงข้อมูลข้อความ **จากไฟล์ผลลัพธ์ข้อมูลสุ่ม 500 แถวที่บันทึกสำเร็จมาจาก Experiment 01 โดยตรง** (ไม่มีการสุ่มสถิติหรือดึงไฟล์ TSV ดิบใหม่) เพื่อให้มั่นใจว่ารันโมเดลทดสอบบนกลุ่มข้อความตัวอย่างที่เหมือนกัน 100%
- **เป้าหมายคำตอบเฉลย (Ground Truth):** ดึงค่าจริง (`text_info` และ `text_human`) จากไฟล์ที่สุ่มบันทึกไว้ใน Exp 01 เพื่อใช้เปรียบเทียบวัดคะแนน F1-score หลังการรันคลาสแบบ 2 เลเยอร์เสร็จสิ้น

---

## 3. การกำหนดค่าโมเดลและสถาปัตยกรรมระบบ (Model & Pipeline Architecture)

ระบบการทดลองจะประยุกต์ใช้โมดูลและไลบรารีในลักษณะเดิม แต่ใช้ Pydantic Schema แบบ 2 เลเยอร์:

### 3.1 การเชื่อมต่อโมเดลผ่าน API
การเชื่อมต่อโมเดลจะใช้ OpenAI SDK ในการเรียกใช้ API ของผู้ให้บริการ ดังนี้:
1. **Gemma 4**:
   - **Endpoint:** `https://openrouter.ai/api/v1`
   - **API Key:** `OPEN_ROUTER_API_KEY`
   - **Model ID:** `google/gemma-4-26b-a4b-it`
2. **deepseek-v4-flash**:
   - **Endpoint:** `https://openrouter.ai/api/v1`
   - **API Key:** `OPEN_ROUTER_API_KEY`
   - **Model ID:** `deepseek/deepseek-v4-flash`
3. **typhoon-v2.5**:
   - **Endpoint:** `https://api.opentyphoon.ai/v1`
   - **API Key:** `TYPHOON_API_KEY`
   - **Model ID:** `typhoon-v2.5-30b-a3b-instruct`

### 3.2 บังคับโครงสร้างผลลัพธ์ด้วย Function Calling (Two-Layer)
ใช้ **Function Calling (Function Call)** ผ่าน Pydantic Schema บังคับผลลัพธ์ในลักษณะของ 2 คีย์แยกกัน เพื่อให้โมเดลประเมินคุณสมบัติทั้งสองของทวีตอย่างอิสระ:

```python
from pydantic import BaseModel, Field
from typing import Literal

class TwoLayerClassificationResult(BaseModel):
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

คำสั่งระบบและคำแนะนำสำหรับผู้ใช้งานจะใช้เป็นภาษาอังกฤษถอดแบบมาจากงานวิจัยดั้งเดิม (ref/disaster_classification.ipynb) 100%:

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

## 5. แผนการวัดผลเปรียบเทียบข้ามสถาปัตยกรรม (Cross-Architecture Evaluation Plan)

เนื่องจากผลลัพธ์ของ Experiment 02 มีข้อมูล 2 เลเยอร์อยู่แล้วในตัวเลือกของผลลัพธ์ จึงสามารถคำนวณวัดผลเปรียบเทียบกับคำทำนายจากเลเยอร์เดียวของ Experiment 01 ได้โดยตรงดังนี้:

1. **การคำนวณเปรียบเทียบ F1-Score:**
   - คำนวณ Informativeness F1 และ Category F1 จากผลทำนายตรงของโมเดล
   - เปรียบเทียบความแม่นยำ F1-Score ระหว่างสถาปัตยกรรม 1-Layer (Exp 1) และ 2-Layer (Exp 2) แยกทีละโมเดล
2. **วิเคราะห์ความสอดคล้อง (Consistency Analysis):**
   - ตรวจสอบว่าโมเดลทำนายได้ตรงตามหลักการหรือไม่ เช่น โมเดลทำนาย `informativeness="not_informative"` คู่กับหมวดหมู่ `category="not_humanitarian"` ได้สอดคล้องกันดีเพียงใด
3. **แผนภูมิวัดประสิทธิภาพเปรียบเทียบ (Performance Chart):**
   - พลอตกราฟบาร์เพื่อสรุปว่า การออกแบบ Prompt แบบแยกสองเลเยอร์ช่วยยกระดับความถูกต้องในการระบุความช่วยเหลือทางมนุษยธรรมขึ้นอย่างมีนัยสำคัญหรือไม่

---

## 6. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จากการรันข้อมูลสุ่มเดิม 500 รายการของ Experiment 02 จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:

```text
e:/nlp-for-disaster/exp2/results/
├── deepseek-v4-flash_results.csv        <- ผลการจำแนก 2 เลเยอร์โดยตรง
├── typhoon-v2.5_results.csv             <- ผลการจำแนก 2 เลเยอร์โดยตรง
├── gemma-4_results.csv                  <- ผลการจำแนก 2 เลเยอร์โดยตรง
├── model_comparison_metrics.csv         <- เปรียบเทียบ F1-Score ระหว่างโมเดลใน Exp 2
├── model_comparison_chart.png           <- กราฟเปรียบเทียบ F1-Score ระหว่างโมเดลใน Exp 2
├── exp1_vs_exp2_comparison.csv          <- ตารางประเมินผลเปรียบเทียบระหว่าง 1-Layer (Exp 1) และ 2-Layer (Exp 2)
└── confusion_matrices/                  <- ภาพ Confusion Matrix แยกรายโมเดลของ Exp 2
    ├── deepseek_confusion_matrices.png
    ├── typhoon_confusion_matrices.png
    └── gemma_confusion_matrices.png
```

### 6.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema)
ไฟล์ผลลัพธ์แยกตามรุ่นโมเดล (`deepseek-v4-flash_results.csv`, `typhoon-v2.5_results.csv`, `gemma-4_results.csv`) สำหรับสถาปัตยกรรม Two-Layer Joint Classification จะจัดเก็บคำทำนายของ AI ทั้งสองเลเยอร์แยกกันในตาราง โดยมีโครงสร้างคอลัมน์ดังนี้:

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `tweet_id` | ไอดีข้อความทวีต (ตรงตามชุดข้อมูลต้นฉบับ) | `8.29177E+17` |
| `tweet_text` | ข้อความภาษาอังกฤษดิบที่ส่งให้โมเดลวิเคราะห์ | *“Red Cross is helping people in Houston...”* |
| `true_text_info` | เฉลยจริง: ความเกี่ยวข้องภัยพิบัติ (Ground Truth) | `informative` / `not_informative` |
| `true_text_human` | เฉลยจริง: หมวดหมู่ช่วยเหลือทางมนุษยธรรม (Ground Truth) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `predicted_info` | คำทำนาย AI เลเยอร์ 1: ความเกี่ยวข้องภัยพิบัติ (ฟิลด์ `informativeness` จาก Function Call) | `informative` / `not_informative` |
| `predicted_category` | คำทำนาย AI เลเยอร์ 2: หมวดหมู่ช่วยเหลือ (ฟิลด์ `category` จาก Function Call) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษ `tweet_text` | `42` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผล | `156` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผล | `22` |
| `latency_seconds` | เวลาในการทำงานแต่ละแถว (หน่วยวินาที) | `1.42` |
