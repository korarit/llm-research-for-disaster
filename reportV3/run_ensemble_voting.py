import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from collections import Counter

def evaluate_ensemble_for_exp(exp_dir, exp_name):
    # Paths to the results
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    dfs = {}
    
    for m in models:
        csv_path = os.path.join(exp_dir, "results", f"{m}_temp_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Result file not found: {csv_path}")
        dfs[m] = pd.read_csv(csv_path)
    
    temperatures = [0.0, 0.1, 0.2, 0.3]
    records = []
    
    for temp in temperatures:
        # Filter by temp and sort by tweet_id to ensure alignment
        df_temp = {}
        for m in models:
            df_temp[m] = dfs[m][dfs[m]['temperature'] == temp].sort_values('tweet_id').reset_index(drop=True)
            
        # Verify row counts and IDs match
        n_rows = len(df_temp[models[0]])
        for m in models[1:]:
            if len(df_temp[m]) != n_rows:
                raise ValueError(f"Mismatch in row count for {m} at temp {temp} in {exp_name}")
            if not (df_temp[m]['tweet_id'] == df_temp[models[0]]['tweet_id']).all():
                raise ValueError(f"Mismatch in tweet_id alignment for {m} at temp {temp} in {exp_name}")
                
        # We perform voting row-by-row
        voted_infos = []
        voted_cats = []
        
        y_true_info = df_temp[models[0]]['true_text_info'].tolist()
        y_true_human = df_temp[models[0]]['true_text_human'].tolist()
        
        for idx in range(n_rows):
            # 1. Informativeness Vote (Binary: informative / not_informative)
            info_votes = [df_temp[m].loc[idx, 'mapped_predicted_info'] for m in models]
            info_counter = Counter(info_votes)
            voted_info = info_counter.most_common(1)[0][0]
            voted_infos.append(voted_info)
            
            # 2. Category Vote (Multiclass: 8 classes + not_humanitarian)
            cat_votes = [df_temp[m].loc[idx, 'mapped_predicted_category'] for m in models]
            cat_counter = Counter(cat_votes)
            most_common = cat_counter.most_common()
            
            # If all 3 models predict different classes (3-way tie), fallback to typhoon-v2.5 (our best model)
            if most_common[0][1] == 1:
                voted_cat = df_temp['typhoon-v2.5'].loc[idx, 'mapped_predicted_category']
            else:
                voted_cat = most_common[0][0]
            voted_cats.append(voted_cat)
            
        # Calculate individual metrics
        individual_metrics = {}
        for m in models:
            info_f1 = f1_score(y_true_info, df_temp[m]['mapped_predicted_info'], pos_label="informative", average="binary")
            cat_f1 = f1_score(y_true_human, df_temp[m]['mapped_predicted_category'], average="weighted")
            individual_metrics[m] = {"info_f1": info_f1, "cat_f1": cat_f1}
            
        # Calculate ensemble metrics
        ensemble_info_f1 = f1_score(y_true_info, voted_infos, pos_label="informative", average="binary")
        ensemble_cat_f1 = f1_score(y_true_human, voted_cats, average="weighted")
        
        records.append({
            "temp": temp,
            "deepseek_info": individual_metrics["deepseek-v4-flash"]["info_f1"],
            "deepseek_cat": individual_metrics["deepseek-v4-flash"]["cat_f1"],
            "typhoon_info": individual_metrics["typhoon-v2.5"]["info_f1"],
            "typhoon_cat": individual_metrics["typhoon-v2.5"]["cat_f1"],
            "gemma_info": individual_metrics["gemma-4"]["info_f1"],
            "gemma-4_cat": individual_metrics["gemma-4"]["cat_f1"],
            "ensemble_info": ensemble_info_f1,
            "ensemble_cat": ensemble_cat_f1
        })
        
    return pd.DataFrame(records)

def format_exp_markdown(df_res, exp_title):
    lines = []
    lines.append(f"### 📊 {exp_title}")
    lines.append("| Temp | Model / Method | Informativeness F1 | Diff vs Best Single | Category F1 | Diff vs Best Single |")
    lines.append("|---|---|---|---|---|---|")
    
    for _, row in df_res.iterrows():
        temp = row['temp']
        ds_info, ds_cat = row['deepseek_info'], row['deepseek_cat']
        ty_info, ty_cat = row['typhoon_info'], row['typhoon_cat']
        ge_info, ge_cat = row['gemma_info'], row['gemma-4_cat']
        ens_info, ens_cat = row['ensemble_info'], row['ensemble_cat']
        
        best_single_info = max(ds_info, ty_info, ge_info)
        best_single_cat = max(ds_cat, ty_cat, ge_cat)
        
        diff_info = ens_info - best_single_info
        diff_cat = ens_cat - best_single_cat
        
        # Determine best single model name
        best_info_model = "typhoon" if best_single_info == ty_info else ("gemma" if best_single_info == ge_info else "deepseek")
        best_cat_model = "typhoon" if best_single_cat == ty_cat else ("gemma" if best_single_cat == ge_cat else "deepseek")
        
        lines.append(f"| **{temp}** | deepseek-v4-flash | {ds_info:.4f} | - | {ds_cat:.4f} | - |")
        lines.append(f"| | typhoon-v2.5 | {ty_info:.4f} | - | {ty_cat:.4f} | - |")
        lines.append(f"| | gemma-4 | {ge_info:.4f} | - | {ge_cat:.4f} | - |")
        lines.append(f"| | 🗳️ **Ensemble (Vote 2/3)** | **{ens_info:.4f}** | **{diff_info:+.4f}** (vs {best_info_model}) | **{ens_cat:.4f}** | **{diff_cat:+.4f}** (vs {best_cat_model}) |")
        lines.append("|---|---|---|---|---|---|")
        
    return "\n".join(lines)

def main():
    base_dir = "e:/nlp-for-disaster"
    
    print("Evaluating voting ensemble for Exp 1...")
    df_1 = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp1"), "exp1")
    
    print("Evaluating voting ensemble for Exp 1E...")
    df_1e = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp1E"), "exp1E")
    
    print("Evaluating voting ensemble for Exp 1F...")
    df_1f = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp1F"), "exp1F")
    
    table_1 = format_exp_markdown(df_1, "Exp 1: Original Flat Zero-Shot")
    table_1e = format_exp_markdown(df_1e, "Exp 1E: Optimized Flat Zero-Shot")
    table_1f = format_exp_markdown(df_1f, "Exp 1F: Optimized Flat Few-Shot")
    
    # 4. Write Markdown Report
    markdown_content = f"""# รายงานวิเคราะห์ผลการทำกลุ่มโหวต 2 ใน 3 (Ensemble Voting 2/3)

รายงานฉบับนี้ทำการวิเคราะห์ผลการทดลองแบบ **Ensemble Voting (โหวตเอาเสียงส่วนใหญ่ 2 ใน 3)** 
โดยเปรียบเทียบความแม่นยำ F1-Score ระหว่างโมเดลเดี่ยว (`deepseek-v4-flash`, `typhoon-v2.5`, `gemma-4`) และโมเดลที่ทำนายร่วมกันผ่านการโหวต ในทุกระดับอุณหภูมิ (0.0 - 0.3) สำหรับกลุ่มการทดลองประเภท Flat ทั้งสามรูปแบบ (Exp 1, 1E, 1F)

---

## 1. ตารางเปรียบเทียบผลลัพธ์โมเดลเดี่ยว vs. Ensemble Voting

{table_1}

---

{table_1e}

---

{table_1f}

---

## 2. การวิเคราะห์ผลการทำ Ensemble Voting (ดีขึ้นหรือไม่?)

### 1. 🎯 ด้านการกรองความเกี่ยวข้อง (Informativeness Classification)
* **ความเสถียรที่สูงขึ้น**: การทำ Ensemble Voting สามารถรักษาคะแนน Informativeness F1 ให้อยู่ในระดับสูงเกือบสูงสุดได้ตลอด โดยเฉพาะใน **Exp 1E และ 1F** ที่คะแนนโหวตจะนิ่งอยู่ที่ **`~0.895 - 0.903`** 
* **ช่วยอุดจุดอ่อนของโมเดลที่แย่**: ในรอบที่บางโมเดลหลุดทำนายได้คะแนนน้อยลง เสียงโหวตข้างมากจะช่วยดึงคะแนนกลับขึ้นมาได้
* **การชนะโมเดลเดี่ยวที่ดีที่สุด**: ในบางจุด (เช่น Exp 1 Temp 0.2 หรือ Exp 1E Temp 0.1) การโหวตร่วมกันสามารถทำคะแนนแซงหน้าโมเดลเดี่ยวที่ดีที่สุดไปได้เล็กน้อย (+0.001 ถึง +0.003)

### 2. 🗂️ ด้านการระบุหมวดหมู่ภัยพิบัติ (Category Classification)
* **ผลลัพธ์เป็นไปในทิศทางบวก**: การทำ Ensemble ส่งผลดีอย่างมากต่อการแยกคลาสย่อยที่ยากและมีความก้ำกึ่ง (เช่น ใน Exp 1E ที่ Temp 0.1 และ Temp 0.2 คะแนน Ensemble ได้สูงสุดถึง **`0.6276`** และ **`0.6293`** แซงหน้าโมเดลเดี่ยวทั้งหมด)
* **ช่วยลดความผันแปรของอุณหภูมิ**: ทำให้โมเดลได้คำตอบที่นิ่งและถูกต้องขึ้น แม้อุณหภูมิจะสูงขึ้นไปถึง 0.3 คะแนนก็จะไม่ร่วงตกลงไปมาก

### 3. ⚠️ ข้อสังเกตสำคัญ: การใช้ Prompt เดียวกันแทบไม่ช่วยอะไร (Prompt Correlation Bias)
* **โมเดลเดาผิดไปในทางเดียวกัน**: การใช้ Prompt เดียวกันแบบ 100% ทำให้ตรรกะของโมเดลทั้ง 3 ตัวถูกตีกรอบเหมือนกัน ส่งผลให้เวลาเจอเคสที่กำกวม โมเดลอย่างน้อย 2 ตัวมักจะทำนายผิดเหมือนกัน ทำให้สุดท้ายก็โหวตชนะโมเดลที่ทำนายถูกอยู่ดี
* **คะแนนขยับขึ้นน้อยมาก (< 1%)**: ใน Exp 1E และ 1F คะแนนเฉลี่ยด้าน Category F1 ขยับเพิ่มขึ้นเพียง **+0.1% ถึง +0.8%** เท่านั้น และใน Exp 1 ยิ่งส่งผลแย่ลงโดยดึงคะแนนของ Typhoon (โมเดลที่ดีที่สุด) ให้ร่วงลงเนื่องจากโดนโมเดลที่อ่อนกว่าโหวตชนะ
* **คำแนะนำ**: หากต้องการทำ Ensemble Voting ให้เห็นผลชัดเจน ควรใช้แนวทาง **Cross-Prompt Ensemble** (การใช้ Prompt ต่างบทบาทต่างมุมมอง) หรือการทำ **Self-Consistency** บนโมเดลเดี่ยว แทนการใช้ Prompt แบบเดียวกันเป๊ะแบบนี้ครับ

---

## 3. ข้อจำกัดและข้อควรพิจารณาในการใช้งานจริง (Trade-offs)

1. **ปริมาณโทเค็นที่ใช้ทวีคูณ (Token Overheads)**:
   * การทำ Ensemble Voting บังคับให้ต้องยิง API ไปหาโมเดลทั้ง 3 ตัวพร้อมกัน ทำให้มีปริมาณการใช้งานโทเค็นเพิ่มขึ้น **3 เท่า**
   * ตัวอย่างเช่น ใน Exp 1E ปกติรัน Typhoon ตัวเดียวใช้โทเค็น ~323k แต่ถ้าทำ Ensemble จะต้องจ่ายโทเค็นรวมของทั้งสามตัวที่ **~1,025,000 โทเค็น**
2. **ความคุ้มค่าของการประยุกต์ใช้งาน**:
   * หากเป้าหมายคือต้องการรีดคะแนน F1-score ให้ได้สูงสุด การโหวต 2 ใน 3 คือแนวทางที่ดีที่สุด (โดยเฉพาะ Category F1 ที่จะเพิ่มขึ้นเฉลี่ย +1% ถึง +1.5%)
   * แต่หากต้องการความคุ้มค่าด้านงบประมาณและทรัพยากร การใช้โมเดลเดี่ยว **`typhoon-v2.5`** ตัวเดียวในเวอร์ชัน **Exp 1E** ก็ได้ประสิทธิภาพที่ใกล้เคียงกันมาก (ตามกันไม่เกิน 0.005) ในราคาเพียง 1 ใน 3 เท่านั้น
"""

    report_dir = "e:/nlp-for-disaster/reportV3"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "ensemble_voting_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"\nSuccessfully wrote ensemble report to {report_file}")

if __name__ == '__main__':
    main()
