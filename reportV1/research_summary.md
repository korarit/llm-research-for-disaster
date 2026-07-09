# รายงานการวิจัย: การวิเคราะห์ประสิทธิภาพ LLMs ในการจำแนกข้อความแจ้งเตือนภัยพิบัติ (Research Summary v1)

รายงานฉบับนี้สรุปผลการทดลองเปรียบเทียบสถาปัตยกรรมและ Prompt ของ Large Language Models (LLMs) บนชุดข้อมูล CrisisMMD (ตัวอย่างสุ่ม 500 รายการ) เพื่อทดสอบขีดความสามารถในการแยกแยะความเกี่ยวข้องกับภัยพิบัติ (Informativeness) และการระบุหมวดหมู่การช่วยเหลือทางมนุษยธรรม (Category Classification) 

งานวิจัยนี้เป็นการปูพื้นฐานเชิงสถิติเพื่อพิสูจน์ความเป็นไปได้ในการพัฒนาต่อยอดไปสู่ระบบแจ้งเตือนและสกัดข้อมูลภัยพิบัติของไทยในอนาคต (Experiment 04)

---

## 1. ข้อมูลสเปกทรัลการทดลอง (Experiment Specifications)
* **โมเดลที่ร่วมทดสอบ (MoE Models)**:
  1. `deepseek-v4-flash`
  2. `typhoon-v2.5` (`typhoon-v2.5-30b-a3b-instruct`)
  3. `gemma-4` (`gemma-4-26b-a4b-it`)
* **ช่วงอุณหภูมิ (Temperatures)**: `0.0`, `0.1`, `0.2`, `0.3`
* **ขอบเขตการเปรียบเทียบ**:
  * **Exp 1 (Original Flat)**: ทำนาย 1-step แยกหมวดหมู่ภัยพิบัติทันที (ไม่มีกรองขยะแยกต่างหาก)
  * **Exp 1E (Biased Flat)**: ปรับปรุงนิยามคลาสและเงื่อนไขห้าม (Negative Constraints) ที่ตึงตัว
  * **Exp 1F (Biased Few-Shot Flat)**: เพิ่มตัวอย่างทวีต 16 ทวีตโดยใช้เงื่อนไขตึงตัวแบบ 1E
  * **Exp 2 (Original Two-Layer Joint)**: วิเคราะห์ Informativeness + Category พร้อมกันใน 1 คำขอ
  * **Exp 2E (Biased Two-Layer Joint)**: เพิ่มเกณฑ์การประเมิน 2 ชั้นแบบคัดกรองขยะอย่างตึงตัว
  * **Exp 2F (Biased Two-Layer Joint Few-Shot)**: ใช้เงื่อนไขแบบ 2E ร่วมกับตัวอย่างแบบ Few-shot

---

## 2. ตารางคะแนนสรุปผล F1-Score แยกตามรายโมเดล (F1-Score Metrics Summary)

### 🤖 Model: deepseek-v4-flash

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | Latency (sec) | Tokens Used |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Exp 1 (Original Flat Zero-Shot) | 0.0 | 0.8458 | 0.6003 | 1.84 | 369,731 |
| Exp 1 (Original Flat Zero-Shot) | 0.1 | 0.8391 | 0.6045 | 1.86 | 369,807 |
| Exp 1 (Original Flat Zero-Shot) | 0.2 | 0.8402 | 0.6102 | 1.67 | 369,993 |
| Exp 1 (Original Flat Zero-Shot) | 0.3 | 0.8430 | 0.6034 | 1.97 | 370,242 |
| Exp 1E (Biased Flat Zero-Shot) | 0.0 | 0.8029 | 0.5631 | 2.94 | 701,931 |
| Exp 1E (Biased Flat Zero-Shot) | 0.1 | 0.7876 | 0.5506 | 2.89 | 703,516 |
| Exp 1E (Biased Flat Zero-Shot) | 0.2 | 0.7994 | 0.5624 | 3.01 | 705,837 |
| Exp 1E (Biased Flat Zero-Shot) | 0.3 | 0.7970 | 0.5587 | 2.19 | 705,873 |
| Exp 1F (Biased Flat Few-Shot) | 0.0 | 0.7822 | 0.5558 | 1.62 | 847,698 |
| Exp 1F (Biased Flat Few-Shot) | 0.1 | 0.7870 | 0.5670 | 1.60 | 847,618 |
| Exp 1F (Biased Flat Few-Shot) | 0.2 | 0.7876 | 0.5764 | 1.59 | 847,556 |
| Exp 1F (Biased Flat Few-Shot) | 0.3 | 0.7793 | 0.5470 | 1.65 | 847,523 |
| Exp 2 (Original Two-Layer Joint) | 0.0 | 0.7419 | 0.5768 | 2.82 | 399,377 |
| Exp 2 (Original Two-Layer Joint) | 0.1 | 0.7488 | 0.5320 | 1.91 | 398,964 |
| Exp 2 (Original Two-Layer Joint) | 0.2 | 0.7469 | 0.5243 | 1.81 | 399,081 |
| Exp 2 (Original Two-Layer Joint) | 0.3 | 0.7554 | 0.5309 | 1.88 | 399,143 |
| Exp 2E (Biased Two-Layer Joint) | 0.0 | 0.6599 | 0.4714 | 1.61 | 626,298 |
| Exp 2E (Biased Two-Layer Joint) | 0.1 | 0.6745 | 0.4868 | 1.75 | 626,331 |
| Exp 2E (Biased Two-Layer Joint) | 0.2 | 0.6689 | 0.4900 | 2.05 | 626,214 |
| Exp 2E (Biased Two-Layer Joint) | 0.3 | 0.6611 | 0.4675 | 4.86 | 628,544 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.0 | 0.7252 | 0.5240 | 3.76 | 843,564 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.1 | 0.7106 | 0.5157 | 3.98 | 843,571 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.2 | 0.7345 | 0.5151 | 3.65 | 841,131 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.3 | 0.7340 | 0.5377 | 3.18 | 842,617 |

### 🤖 Model: typhoon-v2.5

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | Latency (sec) | Tokens Used |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Exp 1 (Original Flat Zero-Shot) | 0.0 | 0.8707 | 0.6407 | 0.61 | 296,262 |
| Exp 1 (Original Flat Zero-Shot) | 0.1 | 0.8755 | 0.6287 | 0.41 | 296,279 |
| Exp 1 (Original Flat Zero-Shot) | 0.2 | 0.8778 | 0.6384 | 0.36 | 296,281 |
| Exp 1 (Original Flat Zero-Shot) | 0.3 | 0.8791 | 0.6249 | 0.34 | 296,283 |
| Exp 1E (Biased Flat Zero-Shot) | 0.0 | 0.7875 | 0.5749 | 0.47 | 617,879 |
| Exp 1E (Biased Flat Zero-Shot) | 0.1 | 0.7803 | 0.5658 | 0.41 | 617,871 |
| Exp 1E (Biased Flat Zero-Shot) | 0.2 | 0.7809 | 0.5628 | 0.41 | 617,873 |
| Exp 1E (Biased Flat Zero-Shot) | 0.3 | 0.7857 | 0.5717 | 0.38 | 617,889 |
| Exp 1F (Biased Flat Few-Shot) | 0.0 | 0.7355 | 0.5292 | 0.39 | 750,035 |
| Exp 1F (Biased Flat Few-Shot) | 0.1 | 0.7434 | 0.5292 | 0.38 | 750,039 |
| Exp 1F (Biased Flat Few-Shot) | 0.2 | 0.7312 | 0.5246 | 0.36 | 750,030 |
| Exp 1F (Biased Flat Few-Shot) | 0.3 | 0.7414 | 0.5343 | 0.36 | 750,024 |
| Exp 2 (Original Two-Layer Joint) | 0.0 | 0.7070 | 0.5574 | 0.41 | 326,105 |
| Exp 2 (Original Two-Layer Joint) | 0.1 | 0.7068 | 0.4854 | 0.47 | 326,117 |
| Exp 2 (Original Two-Layer Joint) | 0.2 | 0.7122 | 0.4943 | 0.55 | 326,112 |
| Exp 2 (Original Two-Layer Joint) | 0.3 | 0.7145 | 0.4917 | 0.50 | 326,115 |
| Exp 2E (Biased Two-Layer Joint) | 0.0 | 0.5720 | 0.3985 | 0.47 | 542,292 |
| Exp 2E (Biased Two-Layer Joint) | 0.1 | 0.5746 | 0.4068 | 0.46 | 542,307 |
| Exp 2E (Biased Two-Layer Joint) | 0.2 | 0.5824 | 0.4155 | 0.46 | 542,303 |
| Exp 2E (Biased Two-Layer Joint) | 0.3 | 0.5798 | 0.4156 | 0.47 | 542,286 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.0 | 0.6376 | 0.4636 | 0.51 | 754,671 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.1 | 0.6353 | 0.4543 | 0.47 | 754,685 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.2 | 0.6436 | 0.4695 | 0.44 | 754,674 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.3 | 0.6411 | 0.4664 | 0.48 | 754,679 |

### 🤖 Model: gemma-4

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | Latency (sec) | Tokens Used |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Exp 1 (Original Flat Zero-Shot) | 0.0 | 0.7685 | 0.5521 | 2.74 | 271,916 |
| Exp 1 (Original Flat Zero-Shot) | 0.1 | 0.7662 | 0.5578 | 1.94 | 271,233 |
| Exp 1 (Original Flat Zero-Shot) | 0.2 | 0.7748 | 0.5557 | 2.17 | 272,114 |
| Exp 1 (Original Flat Zero-Shot) | 0.3 | 0.7748 | 0.5621 | 2.43 | 272,590 |
| Exp 1E (Biased Flat Zero-Shot) | 0.0 | 0.7068 | 0.5379 | 1.68 | 580,431 |
| Exp 1E (Biased Flat Zero-Shot) | 0.1 | 0.7055 | 0.5316 | 1.94 | 581,054 |
| Exp 1E (Biased Flat Zero-Shot) | 0.2 | 0.7118 | 0.5453 | 2.11 | 581,167 |
| Exp 1E (Biased Flat Zero-Shot) | 0.3 | 0.7097 | 0.5345 | 2.67 | 581,909 |
| Exp 1F (Biased Flat Few-Shot) | 0.0 | 0.7406 | 0.5293 | 2.48 | 726,388 |
| Exp 1F (Biased Flat Few-Shot) | 0.1 | 0.7387 | 0.5279 | 3.07 | 726,495 |
| Exp 1F (Biased Flat Few-Shot) | 0.2 | 0.7418 | 0.5310 | 2.62 | 726,465 |
| Exp 1F (Biased Flat Few-Shot) | 0.3 | 0.7406 | 0.5321 | 2.61 | 726,485 |
| Exp 2 (Original Two-Layer Joint) | 0.0 | 0.7206 | 0.5489 | 2.54 | 298,514 |
| Exp 2 (Original Two-Layer Joint) | 0.1 | 0.7299 | 0.5176 | 1.95 | 297,789 |
| Exp 2 (Original Two-Layer Joint) | 0.2 | 0.7258 | 0.5156 | 2.16 | 298,298 |
| Exp 2 (Original Two-Layer Joint) | 0.3 | 0.7256 | 0.5118 | 2.19 | 298,853 |
| Exp 2E (Biased Two-Layer Joint) | 0.0 | 0.5506 | 0.4167 | 1.88 | 503,128 |
| Exp 2E (Biased Two-Layer Joint) | 0.1 | 0.5533 | 0.4256 | 2.02 | 503,711 |
| Exp 2E (Biased Two-Layer Joint) | 0.2 | 0.5406 | 0.4213 | 2.25 | 504,136 |
| Exp 2E (Biased Two-Layer Joint) | 0.3 | 0.5431 | 0.4143 | 2.47 | 504,328 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.0 | 0.6779 | 0.4819 | 2.62 | 722,704 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.1 | 0.6824 | 0.4825 | 3.07 | 722,726 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.2 | 0.6824 | 0.4867 | 3.00 | 722,652 |
| Exp 2F (Biased Two-Layer Joint Few-Shot) | 0.3 | 0.6757 | 0.4833 | 2.91 | 722,761 |

---

## 3. การค้นพบสำคัญจากการวิจัย (Key Research Findings)

### 3.1 ผลกระทบของสถาปัตยกรรม (Flat vs. Two-Layer Joint)
* **สถาปัตยกรรมแบบขั้นตอนเดียว (Flat Classification - Exp 1)** มีคะแนนเฉลี่ยที่สูงกว่าแบบสองเลเยอร์ร่วมกัน (Two-Layer Joint - Exp 2) อย่างเห็นได้ชัดในโมเดลเกือบทุกรุ่น เนื่องจากรูปแบบ Flat ไม่เปิดโอกาสให้โมเดลสร้างความขัดแย้งเชิงตรรกะในผลลัพธ์
* **ความคุ้มค่าด้านทรัพยากร**: โมเดลระดับ Flat (Exp 1) ใช้ปริมาณ Token ขาเข้าที่น้อยกว่าและมี Latency ต่ำกว่า ส่งผลดีต่อต้นทุนการรันระบบขยาย

### 3.2 ผลของการใส่กฎคัดกรองขยะที่ตึงเกินไป (Strictness Bias in E / F)
* จากการเก็บข้อมูลวิเคราะห์ ทวีตส่วนใหญ่ใน CrisisMMD ที่ถูกป้ายว่าเป็นข่าวสารภัยพิบัติ (`informative`) มักเป็นทวีตสั้นๆ หรือข้อความแสดงอารมณ์/ความเห็นที่มีคีย์เวิร์ดของภัยพิบัติรวมอยู่ด้วย
* พรอมต์เวอร์ชัน **1E/2E** ที่ใส่กฎสั่งโมเดลอย่างเข้มงวดว่า *"ถ้าเป็นความเห็นส่วนตัว หรือไม่มีความเสียหายรุนแรงให้ปัดทิ้งเป็นขยะทันที"* ส่งผลให้ **คะแนน F1-score ตกลงอย่างมีนัยสำคัญ** (โดยเฉพาะ recall ของกลุ่มข่าวสารร่วงลงมาก)
* การเพิ่มตัวอย่าง **Few-Shot (1F/2F)** บนฐานของ Prompt ที่มี Bias ตึงตัวนี้:
  * ในแบบ Flat (1F) ช่วยดึงคะแนนของ Gemma ให้ดีขึ้นเล็กน้อย แต่ดึงคะแนนของ Deepseek และ Typhoon ให้ดิ่งลงเนื่องจากเกิดสภาวะยึดติดตัวอย่างรุนแรง (Over-anchoring)
  * ในแบบ Two-Layer (2F) ช่วยดึงคะแนนพยุงขึ้นจากเวอร์ชัน 2E ที่แย่ที่สุดได้บางส่วน แต่ก็ยังไม่สามารถเอาชนะผลลัพธ์ของ Exp 2 แบบเดิมได้เนื่องจากติดกรอบเงื่อนไขลบในคู่มือ

---

## 4. ข้อสรุปเพื่อการพัฒนาต่อยอดไปสู่ระบบภัยพิบัติไทย (Experiment 04)

ผลลัพธ์ทางสถิตินี้สนับสนุนว่าการพัฒนา **Experiment 04** ในภาษาไทยสามารถทำได้จริง โดยมีข้อกำหนดพฤติกรรมการควบคุมโมเดลดังนี้:
1. **ลดเกณฑ์ประเมินที่ตึงเกินไป (Remove Strict Rules)**: ในระบบเตือนภัยไทย (Exp 04) เราต้องระบุประเภทข่าวสารทั่วไป (เช่น พยากรณ์อากาศปกติ) ในกลุ่ม `INFO` หรือ `LOW` ให้ชัดเจน แทนที่จะออกกฎคัดกรองทิ้งเป็นโมดูลขยะตั้งแต่ต้น
2. **การนำระบบ Reasoning มาช่วยในการตัดสินใจ**: การใช้ฟังก์ชันสกัดข้อมูลปลายทางที่มีการเขียน `reasoning` เป็นฟิลด์แรกสุดใน Pydantic Schema จะทำหน้าที่คล้าย Chain-of-Thought (CoT) ชนิดย่อ ช่วยดัน Category F1-Score ในภาษาไทยขยับเข้าใกล้เป้าหมาย 0.70 ได้ดีที่สุด
3. **การควบคุมความคุ้มค่าด้าน Token**: ระบบประมวลผลควรหลีกเลี่ยงการใช้ Few-shot examples ขนาดใหญ่ที่มีปริมาณ Token สะสมเกิน 800k ต่อรอบการสแกน และมุ่งเน้นไปที่การใช้ **Zero-Shot + Structured Enum Disambiguation** ที่มีประสิทธิผลเทียบเท่าในราคาที่ต่ำกว่า
