import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import resample

def get_int_value(val):
    if pd.isna(val) or val is None:
        return 0
    try:
        return int(float(val))
    except:
        return 0

def bootstrap_f1_ci(df_full, model_name, n_iterations=1000, alpha=0.95):
    # Stage 1: Help Request F1
    y_true_s1 = df_full['gt_is_help_request'].astype(bool).values
    y_pred_s1 = df_full['pred_is_help_request'].fillna(False).astype(bool).values
    
    # Filtered dataset for Stage 2 (numeric counts)
    df_stage2 = df_full[df_full['gt_is_help_request'] == True].copy()
    
    # Stage 2 targets
    gts_critical = df_stage2['gt_critical'].apply(get_int_value).values
    preds_critical = df_stage2['pred_victims_critical'].apply(get_int_value).values
    
    gts_urgent = df_stage2['gt_urgent'].apply(get_int_value).values
    preds_urgent = df_stage2['pred_victims_urgent'].apply(get_int_value).values
    
    # Bootstrap lists
    s1_f1_boot = []
    critical_f1_boot = []
    urgent_f1_boot = []
    
    n_size_s1 = len(y_true_s1)
    n_size_s2 = len(df_stage2)
    
    np.random.seed(42)
    for _ in range(n_iterations):
        # Bootstrap Stage 1
        indices_s1 = np.random.choice(n_size_s1, size=n_size_s1, replace=True)
        s1_f1 = f1_score(y_true_s1[indices_s1], y_pred_s1[indices_s1], zero_division=0)
        s1_f1_boot.append(s1_f1)
        
        # Bootstrap Stage 2
        indices_s2 = np.random.choice(n_size_s2, size=n_size_s2, replace=True)
        crit_f1 = f1_score(gts_critical[indices_s2], preds_critical[indices_s2], average='weighted', zero_division=0)
        critical_f1_boot.append(crit_f1)
        
        urg_f1 = f1_score(gts_urgent[indices_s2], preds_urgent[indices_s2], average='weighted', zero_division=0)
        urgent_f1_boot.append(urg_f1)
        
    # Percentiles
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    s1_ci = (np.percentile(s1_f1_boot, lower_p), np.percentile(s1_f1_boot, upper_p))
    crit_ci = (np.percentile(critical_f1_boot, lower_p), np.percentile(critical_f1_boot, upper_p))
    urg_ci = (np.percentile(urgent_f1_boot, lower_p), np.percentile(urgent_f1_boot, upper_p))
    
    # Point estimates
    s1_point = f1_score(y_true_s1, y_pred_s1, zero_division=0)
    crit_point = f1_score(gts_critical, preds_critical, average='weighted', zero_division=0)
    urg_point = f1_score(gts_urgent, preds_urgent, average='weighted', zero_division=0)
    
    return {
        "model": model_name,
        "stage1_f1": (s1_point, s1_ci),
        "critical_f1": (crit_point, crit_ci),
        "urgent_f1": (urg_point, urg_ci)
    }

def main():
    models = ["gemma-4", "deepseek-v4-flash", "typhoon-v2.5"]
    results_dir = "e:/nlp-for-disaster/exp4/results"
    
    for m in models:
        csv_path = os.path.join(results_dir, f"{m}_results.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        stats = bootstrap_f1_ci(df, m)
        print(f"=== Model: {stats['model']} ===")
        print(f"Stage 1 Help Request F1-score: {stats['stage1_f1'][0]:.4f}  95% CI: [{stats['stage1_f1'][1][0]:.4f}, {stats['stage1_f1'][1][1]:.4f}]")
        print(f"Stage 2 Critical F1-score:     {stats['critical_f1'][0]:.4f}  95% CI: [{stats['critical_f1'][1][0]:.4f}, {stats['critical_f1'][1][1]:.4f}]")
        print(f"Stage 2 Urgent F1-score:       {stats['urgent_f1'][0]:.4f}  95% CI: [{stats['urgent_f1'][1][0]:.4f}, {stats['urgent_f1'][1][1]:.4f}]")
        print()

if __name__ == '__main__':
    main()
