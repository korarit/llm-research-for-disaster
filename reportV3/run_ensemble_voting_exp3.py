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
        
        info_col = 'final_predicted_info'
        cat_col = 'final_predicted_category'
        
        for idx in range(n_rows):
            # 1. Informativeness Vote (Binary: informative / not_informative)
            info_votes = [df_temp[m].loc[idx, info_col] for m in models]
            info_counter = Counter(info_votes)
            voted_info = info_counter.most_common(1)[0][0]
            voted_infos.append(voted_info)
            
            # 2. Category Vote (Multiclass: 8 classes + not_humanitarian)
            # Make sure empty/NaN cells in category are treated as not_humanitarian or matching string
            cat_votes = []
            for m in models:
                val = df_temp[m].loc[idx, cat_col]
                if pd.isna(val) or str(val).strip() == "":
                    val = "not_humanitarian"
                cat_votes.append(val)
                
            cat_counter = Counter(cat_votes)
            most_common = cat_counter.most_common()
            
            # If all 3 models predict different classes (3-way tie), fallback to typhoon-v2.5
            if most_common[0][1] == 1:
                voted_cat = df_temp['typhoon-v2.5'].loc[idx, cat_col]
                if pd.isna(voted_cat) or str(voted_cat).strip() == "":
                    voted_cat = "not_humanitarian"
            else:
                voted_cat = most_common[0][0]
            voted_cats.append(voted_cat)
            
        # Calculate individual metrics
        individual_metrics = {}
        for m in models:
            pred_info = df_temp[m][info_col].fillna("not_informative").tolist()
            pred_cat = df_temp[m][cat_col].fillna("not_humanitarian").tolist()
            
            info_f1 = f1_score(y_true_info, pred_info, pos_label="informative", average="binary")
            cat_f1 = f1_score(y_true_human, pred_cat, average="weighted")
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
    
    print("Evaluating voting ensemble for Exp 3...")
    df_3 = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp3"), "exp3")
    
    print("Evaluating voting ensemble for Exp 3E...")
    df_3e = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp3E"), "exp3E")
    
    print("Evaluating voting ensemble for Exp 3F...")
    df_3f = evaluate_ensemble_for_exp(os.path.join(base_dir, "exp3F"), "exp3F")
    
    table_3 = format_exp_markdown(df_3, "Exp 3: Original 2-Agent Sequential")
    table_3e = format_exp_markdown(df_3e, "Exp 3E: Optimized 2-Agent Sequential")
    table_3f = format_exp_markdown(df_3f, "Exp 3F: Optimized 2-Agent Sequential Few-Shot")
    
    output_path = os.path.join(base_dir, "reportV3", "raw_tables_exp3.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- Exp 3 ---\n")
        f.write(table_3)
        f.write("\n\n--- Exp 3E ---\n")
        f.write(table_3e)
        f.write("\n\n--- Exp 3F ---\n")
        f.write(table_3f)
        
    print(f"\nSaved raw tables to {output_path}")

if __name__ == '__main__':
    main()
