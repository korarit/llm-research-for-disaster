# รายงานสรุปผลการทดลองภัยพิบัติภาษาไทยทั้งหมด (Thai Disaster Alert Labeling - Report V4)

รายงานฉบับนี้ทำการรวบรวมและวิเคราะห์ผลการทดลองการคัดแยกข้อความภัยพิบัติภาษาไทยจำนวน 1,000 แถวจากชุดข้อมูล `CrisisMMD_Thai_1000.csv` ภายใต้สถาปัตยกรรมทั้ง 3 รูปแบบ เปรียบเทียบข้ามรุ่นโมเดล (`deepseek-v4-flash`, `typhoon-v2.5`, `gemma-4`) และอุณหภูมิประเมิน `0.0 - 0.3`

---

## 📂 โครงสร้างโฟลเดอร์ผลลัพธ์ใน Report V4

ผลการรันจริงและไฟล์วิเคราะห์แยกตามโฟลเดอร์สถาปัตยกรรมดังนี้:

1. 📂 **[exp1th/](file:///e:/nlp-for-disaster/reportV4/exp1th)** - การจำแนกประเภทแบบ Flat Single-Layer
   - [รายงานสรุปของ Exp 1TH](file:///e:/nlp-for-disaster/reportV4/exp1th/summary.md)
   - [ตารางค่าสถิติ F1 ทุกโมเดล](file:///e:/nlp-for-disaster/reportV4/exp1th/th_model_comparison_metrics.csv)
   - [แผนภูมิเปรียบเทียบภาพรวม F1](file:///e:/nlp-for-disaster/reportV4/exp1th/th_model_comparison_chart.png)
2. 📂 **[exp2th/](file:///e:/nlp-for-disaster/reportV4/exp2th)** - การจำแนกประเภทแบบ Two-Layer Joint
   - [รายงานสรุปของ Exp 2TH](file:///e:/nlp-for-disaster/reportV4/exp2th/summary.md)
   - [ตารางเปรียบเทียบเชิงสถิติ Exp 1TH vs. Exp 2TH](file:///e:/nlp-for-disaster/reportV4/exp2th/th_exp1_vs_exp2_comparison.csv)
3. 📂 **[exp3th/](file:///e:/nlp-for-disaster/reportV4/exp3th)** - การจำแนกประเภทแบบ 2-Agent Sequential Pipeline
   - [รายงานสรุปของ Exp 3TH](file:///e:/nlp-for-disaster/reportV4/exp3th/summary.md)
   - [ตารางเปรียบเทียบรวมทั้ง 3 สถาปัตยกรรม](file:///e:/nlp-for-disaster/reportV4/exp3th/th_exp3_vs_other_comparison.csv)

📝 **[รายงานวิเคราะห์เหตุผลที่ภาษาไทยชนะภาษาอังกฤษ (why_th_better_en.md)](file:///e:/nlp-for-disaster/reportV4/why_th_better_en.md)**

---

## 🏆 โมเดลและสถาปัตยกรรมที่ได้ประสิทธิภาพดีที่สุด (Best Performance)

### 1. หมวดหมู่ผู้ได้รับผลกระทบย่อย (Category F1 Score)
- **อันดับ 1**: **`deepseek-v4-flash`** บนโครงสร้าง **2-Agent (Exp 3TH)** ได้คะแนนสูงสุด **`0.7583`** (T=0.3)
- **อันดับ 2**: **`gemma-4`** บนโครงสร้าง **Flat (Exp 1TH)** ได้คะแนนสูงสุด **`0.7557`** (T=0.1)
- **อันดับ 3**: **`typhoon-v2.5`** บนโครงสร้าง **2-Agent (Exp 3TH)** ได้คะแนนสูงสุด **`0.7403`** (T=0.3)

### 2. หมวดหมู่ความเกี่ยวข้องภัยพิบัติ (Informativeness F1 Score)
- ทุกสถาปัตยกรรมและทุกโมเดลทำคะแนนได้ดีเสถียรมากอยู่ในช่วง **`0.896 - 0.913`** โดยโมเดลที่กรองความเกี่ยวข้องได้แม่นยำที่สุดคือ **`deepseek-v4-flash`** และ **`typhoon-v2.5`**

---

## 💡 ข้อเสนอแนะและข้อสรุปทางวิศวกรรม (Engineering Takeaways)

1. **การปรับแต่งลำดับขั้นตอน (Agent-based Design)**:
   - สถาปัตยกรรมแบบ **2-Agent Sequential (Exp 3TH)** มีการคัดกรองสัญญาณในขั้นแรกออกก่อน ช่วยให้เอเจนต์ตัวที่ 2 สามารถจำแนกความต้องการย่อยของภาษาไทยได้ดียิ่งขึ้นอย่างก้าวกระโดด โดยเฉพาะ **Typhoon-v2.5** (F1 เพิ่มขึ้นจาก `0.68` เป็น `0.74`) และ **DeepSeek** (F1 เพิ่มขึ้นจาก `0.752` เป็น `0.758`)
2. **ทางเลือกสำหรับระบบการนำไปใช้งานจริง (Production Trade-offs)**:
   - **กรณีต้องการความรวดเร็วและคุ้มค่าสูงที่สุด**: แนะนำให้ใช้ **`typhoon-v2.5` บนสถาปัตยกรรม 2-Agent (Exp 3TH)** เนื่องจากทำงานเร็วมาก (Latency เพียง ~0.67 วินาที) โดยมี F1 Category สูงถึง `0.7403`
   - **กรณีต้องการความแม่นยำสูงสุดโดยไม่กังวลความเร็ว**: แนะนำให้ใช้ **`deepseek-v4-flash` บนสถาปัตยกรรม 2-Agent (Exp 3TH)** ซึ่งจะทำงานช้ากว่า (Latency ~3.04s) แต่จะได้ความถูกต้องสูงสุดที่ `0.7583`
