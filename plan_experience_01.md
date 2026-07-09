# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Disaster Alert Labeling Experiment - 01)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 01** ซึ่งเป็นการคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **เลเยอร์เดียวขั้นตอนเดียว (Single-Layer / Flat Classification)** อ้างอิงโครงสร้างข้อมูลจากงานวิจัย [Zero-Shot Social Media Crisis Classification: A Training-Free Multimodal Approach](https://digibug.ugr.es/bitstream/handle/10481/111587/applsci-16-02192.pdf?sequence=1&isAllowed=y) (ซอร์สโค้ดต้นฉบับ: [disaster_classification.ipynb](file:///e:/nlp-for-disaster/ref/disaster_classification.ipynb))

ในการทดลองที่ 1 นี้ จะปรับเปลี่ยนแนวทางใน 3 ส่วนสำคัญ:
1. **โมเดลประมวลผล (LLM Models):** เปลี่ยนจากโมเดล Mistral-Small GGUF บน Local Llama Server ไปเป็นโมเดล Open-Source กลุ่ม Mixture of Experts (MoE) 3 รุ่น เพื่อทดสอบและเปรียบเทียบประสิทธิภาพ ได้แก่:
   - **deepseek-v4-flash** (ใช้ OpenRouter `deepseek/deepseek-v4-flash`)
   - **typhoon-v2.5** (ใช้ OpenTyphoon `typhoon-v2.5-30b-a3b-instruct`)
   - **Gemma 4** (ใช้ OpenRouter `google/gemma-4-26b-a4b-it`)
2. **รูปแบบสถาปัตยกรรม (Single-Layer / Flat):** ใช้ขั้นตอนประมวลผลข้อความเพียงระดับเดียว โดยให้โมเดลตอบกลับเฉพาะข้อมูลหมวดหมู่ย่อยร่วมกับระบุความไม่เกี่ยวข้องดิบจบในฟิลด์เดียว (Text-Only Mode)
3. **ขนาดการทดสอบ (Sample Size):** กำหนดการสุ่มข้อมูลการทดสอบไว้ที่ **500 รายการ** จากชุดข้อมูลทั้งหมดเพื่อการทดสอบที่มีประสิทธิภาพและควบคุมค่าใช้จ่าย
4. **อุณหภูมิการทำงาน (Temperatures):** ประเมินผลเพิ่มเติมที่อุณหภูมิ **0.1, 0.2, 0.3** ควบคู่ไปกับค่าเริ่มต้น **0.0**

---

## 1. วัตถุประสงค์การทดลอง (Objectives)
- ประเมินขีดความสามารถการทำ Zero-shot classification แบบเลเยอร์เดียว (Single-Layer) ของกลุ่มโมเดล MoE ทั้ง 3 รุ่น บนข้อมูลข้อความทวีตภาษาอังกฤษจากชุดข้อมูลภัยพิบัติสากล CrisisMMD_v2.0
- ทดสอบความคล่องตัวของการประมวลผลประเภทข้อมูลแบบขั้นตอนเดียวแบนราบ (Flat) เพื่อใช้เป็นฐานข้อมูลหลัก (Baseline) ในการทดสอบสถาปัตยกรรมระดับถัดไป
- บันทึกการคำนวณผลลัพธ์ความแม่นยำ F1-Score ในระดับข้อความ เพื่อใช้เปรียบเทียบในอนาคต

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล (Dataset & Preparation)
สคริปต์การทดลองจะประมวลผลไฟล์คำอธิบาย (.tsv) ภายใต้ชุดข้อมูล `CrisisMMD_v2.0` โดยมีรายละเอียดดังนี้:

- **ที่อยู่ชุดข้อมูล (Dataset Location):** `e:/nlp-for-disaster/dataset/crisis-mmd/`
- **โครงสร้างข้อมูลเข้าหลัก (Input Columns):**
  - `tweet_id`: ไอดีข้อความทวีต
  - `tweet_text`: ข้อความภาษาอังกฤษดิบที่ได้จากสื่อสังคมออนไลน์
- **เป้าหมายคำตอบเฉลย (Ground Truth Columns):**
  - `text_info`: ระบุว่าข้อความเกี่ยวข้องกับภัยพิบัติหรือไม่ (`informative` / `not_informative`)
  - `text_human`: ระบุหมวดหมู่การช่วยเหลือทางมนุษยธรรม
- **การสุ่มตัวอย่าง (Sampling Strategy):**
  - สคริปต์จะอ่านไฟล์ TSV ทั้งหมด คัดกรองบรรทัดข้อมูลที่ไม่มีค่าว่างในคอลัมน์สำคัญ
  - ทำการ **สุ่มตัวอย่างรวมจำนวน 500 แถว** โดยกระจายตัวอย่างอย่างเหมาะสมตามกลุ่มภัยพิบัติต่าง ๆ เพื่อทำหน้าที่เป็นชุดข้อมูลประเมินผลกลางในการทดลองนี้

---

## 3. การกำหนดค่าโมเดลและสถาปัตยกรรมระบบ (Model & Pipeline Architecture)

ระบบการทำทดลองจะเขียนขึ้นด้วยภาษา Python โดยประยุกต์ใช้โมดูลและไลบรารีดังนี้:

### 3.1 การเชื่อมต่อโมเดลผ่าน API
การเชื่อมต่อโมเดลจะใช้ OpenAI SDK ในการเรียกใช้ API ของผู้ให้บริการ ดังนี้:
1. **Gemma 4**:
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("OPEN_ROUTER_API_KEY")`
   - **Endpoint:** `https://openrouter.ai/api/v1`
   - **Model ID:** `google/gemma-4-26b-a4b-it`
2. **deepseek-v4-flash**:
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("OPEN_ROUTER_API_KEY")`
   - **Endpoint:** `https://openrouter.ai/api/v1`
   - **Model ID:** `deepseek/deepseek-v4-flash`
3. **typhoon-v2.5**:
   - **ไลบรารี:** `openai`
   - **การกำหนดสิทธิ์:** คีย์จาก `os.environ.get("TYPHOON_API_KEY")`
   - **Endpoint:** `https://api.opentyphoon.ai/v1`
   - **Model ID:** `typhoon-v2.5-30b-a3b-instruct`

### 3.2 บังคับโครงสร้างผลลัพธ์ด้วย Function Calling (Single-Layer)
ใช้ความสามารถของ SDK ในการทำ **Function Calling (Function Call)** โดยแปลง Pydantic Class ให้เป็นคำจำกัดความเครื่องมือ (Tool/Function Definition) เพื่อบังคับให้ตอบหมวดหมู่ที่เหมาะสมที่สุดเพียงฟิลด์เดียว โดยยุบรวมกรณีที่ข้อความไม่เกี่ยวข้องให้มีค่าเป็น `not_informative`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SingleLayerClassificationResult(BaseModel):
    category: Literal[
        "not_informative",  # ยุบรวมข้อความขยะ/คำบ่นทั่วไป/สิ่งที่ไม่เกี่ยวกับภัยพิบัติไว้ที่นี่
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "injured_or_dead_people",
        "missing_or_found_people",
        "other_relevant_information",
        "rescue_volunteering_or_donation_effort",
        "vehicle_damage"
    ] = Field(description="เลือกหมวดหมู่ที่ระบุรายละเอียดภัยพิบัติเด่นชัดที่สุดเพียงข้อเดียว หรือระบุว่าไม่เกี่ยวข้อง/ไม่มีข้อมูลที่เป็นประโยชน์")
```

### 3.3 การประมวลผลขนานและควบคุมขีดจำกัด (Rate Limit Handling)
- ใช้ `ThreadPoolExecutor` ในการส่ง Request แบบคู่ขนาน
- เสริมระบบ **Exponential Backoff Retry** ครอบคำสั่งการส่ง API เสมอ ในกรณีที่เกิดปัญหา `429 ResourceExhausted` เพื่อความเสถียร

---

## 4. การออกแบบคำสั่งสำหรับโมเดลแบบเลเยอร์เดียว (Single-Layer Prompt Design)

การออกแบบคำสั่งเขียนอธิบายหมวดหมู่ทั้งหมดให้โมเดลเลือกตัดสินใจจากตัวเลือกที่เป็นอิสระต่อกัน (Mutually Exclusive Labels):

### 4.1 คำสั่งระบบ (System Instruction)
```markdown
You are an expert humanitarian disaster analyst with extensive experience in classifying disaster-related content. Your task is to classify tweets into a single specific category based on objective evidence rather than emotional responses.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Tweet: "{tweet_text}"

CLASSIFICATION CRITERIA:
Classify the tweet into exactly ONE of the following categories. Choose the most specific and dominant category represented in the text:

- not_informative: The tweet does NOT contain specific information, represents only emotions, prayers, general opinions, political comments, jokes, or is completely unrelated to disaster management.
- affected_individuals: Mentions displaced people, survivors, or evacuees who are affected but does NOT report deaths or injuries.
- infrastructure_and_utility_damage: References damaged buildings, roads, bridges, electricity, water lines, or other utilities.
- injured_or_dead_people: Reports specific numbers or accounts of injured, hospitalized, or deceased individuals.
- missing_or_found_people: Mentions people who are currently missing, search/rescue missions looking for individuals, or people who have been found/rescued.
- rescue_volunteering_or_donation_effort: Mentions relief goods, donation drives, financial aid, volunteer networks, or rescue team deployment.
- vehicle_damage: References damaged cars, trucks, buses, trains, or rescue vehicles.
- other_relevant_information: General informative reports such as weather forecasts, storm paths, satellite observations, or warnings without specific human or physical impact details.

Return classification by calling the specified function.
```

---

## 5. แผนการวัดผลและการแปลงข้อมูลกลับ (Evaluation & Post-Processing Plan)

เนื่องจากชุดข้อมูล CrisisMMD ติดสลาก Ground Truth แยกประเภทเป็น 2 ฟิลด์คู่ขนาน เพื่อให้เราประเมินเปรียบเทียบกับงานวิจัยเดิมได้ สคริปต์วิเคราะห์จะทำการแปลงผลลัพธ์ (Mapping) ย้อนกลับดังนี้:

### 5.1 ตารางการแมปคลาส (Mapping Rules)
| ผลลัพธ์ที่ทำนายได้จาก Single-Layer | ข้อมูลเฉลย Informativeness ที่จะบันทึกเทียบ | ข้อมูลเฉลย Category ที่จะบันทึกเทียบ |
| :--- | :--- | :--- |
| `not_informative` | `not_informative` | `not_humanitarian` |
| `affected_individuals` | `informative` | `affected_individuals` |
| `infrastructure_and_utility_damage` | `informative` | `infrastructure_and_utility_damage` |
| `injured_or_dead_people` | `informative` | `injured_or_dead_people` |
| `missing_or_found_people` | `informative` | `missing_or_found_people` |
| `other_relevant_information` | `informative` | `other_relevant_information` |
| `rescue_volunteering_or_donation_effort` | `informative` | `rescue_volunteering_or_donation_effort` |
| `vehicle_damage` | `informative` | `vehicle_damage` |

### 5.2 ตัวชี้วัดที่คำนวณ
- **Text Informativeness F1:** คำนวณแบบ Binary F1 (เน้นกลุ่ม `informative`) หลังผ่านการแมปย้อนกลับ
- **Text Category F1:** คำนวณแบบ Weighted F1 เปรียบเทียบกับข้อมูล Category ต้นฉบับ
- **Confusion Matrix:** สร้างแยกของแต่ละหมวดหมู่และบันทึกเป็นรูปภาพ PNG

---

## 6. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จากการสุ่ม 500 รายการของ Experiment 01 จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:

```text
e:/nlp-for-disaster/exp1/results/
├── deepseek-v4-flash_results.csv        <- ผลการจำแนกและผลลัพธ์ที่แมปแล้ว
├── typhoon-v2.5_results.csv             <- ผลการจำแนกและผลลัพธ์ที่แมปแล้ว
├── gemma-4_results.csv                  <- ผลการจำแนกและผลลัพธ์ที่แมปแล้ว
├── model_comparison_metrics.csv         <- เปรียบเทียบ F1-Score ระหว่างโมเดลใน Exp 1
├── model_comparison_chart.png           <- กราฟเปรียบเทียบ F1-Score ระหว่างโมเดลใน Exp 1
└── confusion_matrices/                  <- ภาพ Confusion Matrix แยกรายโมเดลของ Exp 1
    ├── deepseek_confusion_matrices.png
    ├── typhoon_confusion_matrices.png
    └── gemma_confusion_matrices.png
```

### 6.1 โครงสร้างของไฟล์ CSV ผลลัพธ์รายโมเดล (Individual Model CSV Schema)
ไฟล์ผลลัพธ์แยกตามรุ่นโมเดล (`deepseek-v4-flash_results.csv`, `typhoon-v2.5_results.csv`, `gemma-4_results.csv`) จะจัดเก็บคำทำนาย (Label) ของ AI คู่ขนานไปกับข้อความนำเข้าและเฉลยจริง โดยประกอบไปด้วยคอลัมน์ดังนี้:

| ชื่อคอลัมน์ (Column Name) | คำอธิบาย (Description) | ตัวอย่างข้อมูล (Example) |
| :--- | :--- | :--- |
| `tweet_id` | ไอดีข้อความทวีต (ตรงตามชุดข้อมูลต้นฉบับ) | `8.29177E+17` |
| `tweet_text` | ข้อความภาษาอังกฤษดิบที่ส่งให้โมเดลวิเคราะห์ | *“Red Cross is helping people in Houston...”* |
| `true_text_info` | เฉลยจริง: ความเกี่ยวข้องภัยพิบัติ (Ground Truth) | `informative` / `not_informative` |
| `true_text_human` | เฉลยจริง: หมวดหมู่ช่วยเหลือทางมนุษยธรรม (Ground Truth) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `predicted_category` | ผลการคัดแยกโดยตรงที่ AI ตอบกลับมาตาม Function Calling (Single-Layer) | `rescue_volunteering_or_donation_effort` / `not_informative` |
| `mapped_predicted_info` | ผลความเกี่ยวข้อง (Informativeness) ที่ได้หลังการแปลงผล (Mapping) | `informative` / `not_informative` |
| `mapped_predicted_category` | ผลหมวดหมู่ช่วยเหลือ (Category) ที่ได้หลังการแปลงผล (Mapping) | `rescue_volunteering_or_donation_effort` / `not_humanitarian` |
| `tweet_text_char_count` | จำนวนตัวอักษรของข้อความภาษาอังกฤษ `tweet_text` | `42` |
| `token_in_use` | จำนวน Token ขาเข้าที่ใช้ประมวลผล | `156` |
| `token_out_use` | จำนวน Token ขาออกที่ใช้ประมวลผล | `12` |
| `latency_seconds` | เวลาในการทำงานแต่ละแถว (หน่วยวินาที) | `1.42` |
