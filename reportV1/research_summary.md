# สรุปผลการทดลอง: เปรียบเทียบประสิทธิภาพ LLM ในการคัดแยกข้อความภัยพิบัติ (Research Summary v1)

เอกสารฉบับนี้สรุปผลการทดลองเปรียบเทียบสถาปัตยกรรมระบบการคัดแยกข้อความและรูปแบบ Prompt ของ Large Language Models (LLM) บนชุดข้อมูล CrisisMMD (ใช้ตัวอย่างสุ่ม 500 รายการ) เพื่อทดสอบความสามารถในการกรองข้อความเกี่ยวข้องกับภัยพิบัติ (Informativeness) และการระบุหมวดหมู่ช่วยเหลือทางมนุษยธรรม (Category Classification) 

เป้าหมายของการทดลองนี้คือการหาแนวทางเชิงเทคนิคที่ดีที่สุด ก่อนจะนำไปพัฒนาต่อยอดเป็นระบบแจ้งเตือนและสกัดข้อมูลภัยพิบัติเวอร์ชันภาษาไทยในขั้นตอนถัดไป (Experiment 04)

---

## 1. รายละเอียดและรูปแบบการทดลอง (Experiment Settings)
* **โมเดลที่ร่วมทดสอบ (MoE Models)**:
  1. `deepseek-v4-flash`
  2. `typhoon-v2.5` (`typhoon-v2.5-30b-a3b-instruct`)
  3. `gemma-4` (`gemma-4-26b-a4b-it`)
* **ช่วงอุณหภูมิ (Temperatures)**: `0.0`, `0.1`, `0.2`, `0.3`
* **สถาปัตยกรรมและ Prompt ที่ใช้ทดสอบ**:
  * **Exp 1 (Original Flat)**: โมเดลตัดสินใจแบบขั้นตอนเดียว (Single-step) เพื่อแยกหมวดหมู่ภัยพิบัติทันที โดยไม่มีการแยกตัวกรองออกมาก่อน
  * **Exp 1E (Biased Flat)**: ปรับปรุง Prompt ให้ละเอียดขึ้น ขยายนิยามของแต่ละคลาส และเพิ่มกฎเหล็ก/ข้อห้าม (Negative Constraints) ที่เข้มงวด
  * **Exp 1F (Biased Few-Shot Flat)**: ใช้ Prompt และกฎเกณฑ์แบบ 1E ร่วมกับการใส่ตัวอย่างทวีตประกอบ (Few-shot) จำนวน 16 ตัวอย่าง
  * **Exp 2 (Original Two-Layer Joint)**: ให้โมเดลประเมินความเกี่ยวข้องภัยพิบัติ (Layer 1) และหมวดหมู่ช่วยเหลือ (Layer 2) พร้อมกันในการเรียก API ครั้งเดียว (คืนค่ากลับมาเป็น JSON)
  * **Exp 2E (Biased Two-Layer Joint)**: ใช้ระบบตัดสินใจ 2 ชั้นร่วมกับ Prompt แบบเพิ่มเกณฑ์การกรองอย่างเข้มงวด
  * **Exp 2F (Biased Two-Layer Joint Few-Shot)**: ใช้ระบบ 2 ชั้นร่วมกับ Prompt แบบเข้มงวด (2E) และใส่ตัวอย่าง Few-shot

---

## 2. ตารางคะแนนสรุปผล F1-Score แยกตามโมเดล (F1-Score Metrics Summary)

### 🤖 Model: deepseek-v4-flash

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | เวลาเฉลี่ย/แถว (วินาที) | Tokens Used |
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
| Exp 3 (Original 2-Agent) | 0.0 | 0.7917 | 0.5466 | 3.18 | 444,458 |
| Exp 3 (Original 2-Agent) | 0.1 | 0.7917 | 0.5530 | 4.21 | 444,087 |
| Exp 3 (Original 2-Agent) | 0.2 | 0.7935 | 0.5629 | 3.24 | 444,686 |
| Exp 3 (Original 2-Agent) | 0.3 | 0.7876 | 0.5605 | 3.29 | 444,870 |
| Exp 3E (Biased 2-Agent) | 0.0 | 0.8354 | 0.5571 | 4.78 | 677,202 |
| Exp 3E (Biased 2-Agent) | 0.1 | 0.8243 | 0.5139 | 5.52 | 686,699 |
| Exp 3E (Biased 2-Agent) | 0.2 | 0.8602 | 0.5737 | 5.11 | 692,773 |
| Exp 3E (Biased 2-Agent) | 0.3 | 0.8308 | 0.5612 | 4.26 | 670,751 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.0 | 0.8364 | 0.6076 | 5.25 | 900,955 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.1 | 0.8381 | 0.6054 | 4.19 | 901,559 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.2 | 0.8430 | 0.5997 | 5.86 | 902,155 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.3 | 0.8317 | 0.5947 | 6.63 | 912,169 |


### 🤖 Model: typhoon-v2.5

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | เวลาเฉลี่ย/แถว (วินาที) | Tokens Used |
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
| Exp 3 (Original 2-Agent) | 0.0 | 0.6899 | 0.4831 | 0.53 | 303,029 |
| Exp 3 (Original 2-Agent) | 0.1 | 0.6997 | 0.4930 | 0.53 | 304,510 |
| Exp 3 (Original 2-Agent) | 0.2 | 0.6922 | 0.4772 | 0.48 | 302,085 |
| Exp 3 (Original 2-Agent) | 0.3 | 0.6932 | 0.4802 | 0.49 | 303,085 |
| Exp 3E (Biased 2-Agent) | 0.0 | 0.8235 | 0.5600 | 0.61 | 551,556 |
| Exp 3E (Biased 2-Agent) | 0.1 | 0.8273 | 0.5634 | 0.65 | 554,848 |
| Exp 3E (Biased 2-Agent) | 0.2 | 0.8324 | 0.5645 | 0.58 | 553,225 |
| Exp 3E (Biased 2-Agent) | 0.3 | 0.8312 | 0.5607 | 0.57 | 554,022 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.0 | 0.7607 | 0.5454 | 0.52 | 713,517 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.1 | 0.7565 | 0.5528 | 0.51 | 714,742 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.2 | 0.7709 | 0.5555 | 0.51 | 722,189 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.3 | 0.7676 | 0.5589 | 0.48 | 715,991 |


### 🤖 Model: gemma-4

| การทดลอง (Experiment) | Temp | Informativeness F1 | Category F1 | เวลาเฉลี่ย/แถว (วินาที) | Tokens Used |
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
| Exp 3 (Original 2-Agent) | 0.0 | 0.7729 | 0.5344 | 4.15 | 275,053 |
| Exp 3 (Original 2-Agent) | 0.1 | 0.7704 | 0.5424 | 2.25 | 271,562 |
| Exp 3 (Original 2-Agent) | 0.2 | 0.7652 | 0.5446 | 2.47 | 268,825 |
| Exp 3 (Original 2-Agent) | 0.3 | 0.7746 | 0.5431 | 2.31 | 271,752 |
| Exp 3E (Biased 2-Agent) | 0.0 | 0.8143 | 0.5780 | 2.00 | 470,832 |
| Exp 3E (Biased 2-Agent) | 0.1 | 0.8165 | 0.5774 | 2.31 | 473,047 |
| Exp 3E (Biased 2-Agent) | 0.2 | 0.8148 | 0.5792 | 2.52 | 471,672 |
| Exp 3E (Biased 2-Agent) | 0.3 | 0.8126 | 0.5725 | 2.42 | 468,415 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.0 | 0.8411 | 0.6052 | 2.18 | 726,797 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.1 | 0.8387 | 0.6081 | 2.16 | 729,693 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.2 | 0.8371 | 0.6133 | 2.56 | 727,059 |
| Exp 3F (Biased 2-Agent Few-Shot) | 0.3 | 0.8383 | 0.6029 | 2.28 | 727,261 |


---

## 3. ข้อค้นพบสำคัญจากการทดลอง (Key Research Findings)

### 3.1 ผลกระทบของการเลือกใช้สถาปัตยกรรม (Flat vs. Two-Layer Joint)
* **โมเดลแบบขั้นตอนเดียว (Flat Classification - Exp 1)** ทำคะแนนเฉลี่ยได้สูงกว่าแบบทำนายพร้อมกันสองเลเยอร์ (Two-Layer Joint - Exp 2) อย่างเห็นได้ชัดในเกือบทุกโมเดล เนื่องจากสถาปัตยกรรมแบบขั้นตอนเดียวช่วยลดความสับสนในกรณีที่ต้องกรองข้อมูลหลายเรื่องในคำขอเดียว
* **ความคุ้มค่าและ Latency**: โมเดลในกลุ่ม Flat (Exp 1) ใช้โทเคนขาเข้าน้อยกว่าและใช้เวลาประมวลผลเร็วกว่ามาก ส่งผลให้ประหยัดงบประมาณและทรัพยากรเมื่อนำไปใช้จริงในระยะยาว

### 3.2 ผลของการบีบกฎเกณฑ์ให้เข้มงวดเกินไป (Strictness Bias ในกลุ่ม E / F)
* จากการวิเคราะห์ข้อมูลดิบ ทวีตภัยพิบัติส่วนใหญ่ใน CrisisMMD ที่เป็นประโยชน์ต่อการรายงานภัยพิบัติ (`informative`) มักเป็นทวีตสั้นๆ หรือทวีตแนวแสดงอารมณ์/ความคิดเห็นที่แทรกข้อมูลความเสียหายอยู่
* การออกแบบ Prompt ในเวอร์ชัน **1E และ 2E** ที่บังคับโมเดลอย่างตึงตัวว่า *"หากเป็นความเห็นส่วนตัวหรือคำอธิษฐานที่ไม่มีการระบุความเสียหายอย่างชัดเจน ให้ปัดตกเป็นขยะทันที"* ส่งผลให้ **คะแนน F1-score ตกฮวบอย่างรุนแรง** (โดยค่า Recall ของการกรองความเกี่ยวข้องร่วงลงอย่างมีนัยสำคัญ เนื่องจากโมเดลกลัวและเลือกคัดกรองขยะเยอะเกินไป)
* การเสริมตัวอย่าง **Few-Shot (1F / 2F)** เข้ามาบนฐานของ Prompt ที่ปรับกฎตึงตัวแล้วนี้:
  * ในระบบ Flat (1F) ช่วยพยุงคะแนนของโมเดล Gemma ได้บ้าง แต่กลับทำให้ DeepSeek และ Typhoon คะแนนแย่ลงไปอีก เพราะเกิดอาการยึดติดแพทเทิร์นของตัวอย่าง (Over-anchoring)
  * ในระบบ Two-Layer (2F) ช่วยดึงคะแนนให้ดีขึ้นจากเวอร์ชัน 2E แต่ก็ยังแย่กว่าการรันแบบปกติใน Exp 2 อยู่ดี

---

## 4. ข้อเสนอแนะเพื่อพัฒนาต่อยอดระบบภัยพิบัติภาษาไทย (Experiment 04)

ข้อมูลจากการทดลองนี้ชี้ให้เห็นแนวทางที่ควรนำไปปรับใช้กับการทดลองภาษาไทย (Exp 04) ดังนี้:
1. **หลีกเลี่ยงการใช้กฎกรองทิ้งที่เข้มงวดจนเกินไป (Avoid Over-filtering)**: ในระบบภาษาไทย เราควรขยายขอบเขตให้ครอบคลุมข่าวสารทั่วไป (เช่น คำเตือนพยากรณ์อากาศปกติ) โดยจัดกลุ่มให้อยู่ในประเภทความเสี่ยงระดับต่ำ (Low Risk) แทนที่จะพยายามตั้งกฎเหล็กเพื่อปัดตกตั้งแต่ขั้นตอนการกรองขั้นแรก
2. **เพิ่มฟิลด์สำหรับสรุปเหตุผลเชิงตรรกะ (Reasoning-based Extraction)**: การสกัดข้อมูลปลายทางที่มีฟิลด์ `reasoning` เป็นคีย์แรกใน Pydantic Schema (ทำหน้าที่เสมือน Chain-of-Thought แบบสั้น) จะช่วยยกระดับ F1-score ในหมวดหมู่ช่วยเหลือทางมนุษยธรรมให้เข้าใกล้เป้าหมาย 0.70 ได้ดีที่สุด
3. **ควบคุมขนาดและโครงสร้างของ Prompt**: ควรเลี่ยงการใส่ Few-shot examples ขนาดใหญ่ที่มีสิทธิ์ดันจำนวนโทเคนต่อการรันพุ่งสูงเกินไป และหันมาใช้วิธี **Zero-Shot + Structured Enum Disambiguation** (การแบ่งข้อจำกัดและแจกแจง Enum อย่างเป็นระบบ) ซึ่งให้ประสิทธิภาพดีใกล้เคียงกันแต่ประหยัดค่าใช้จ่ายได้มากกว่า
