import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def clean_phone(phone_val):
    if pd.isna(phone_val) or phone_val is None or str(phone_val).strip().lower() in ['none', 'null', 'nan', '']:
        return None
    # Remove all non-digit characters
    return re.sub(r'\D', '', str(phone_val))

def clean_text_field(val):
    if pd.isna(val) or val is None or str(val).strip().lower() in ['none', 'null', 'nan', '']:
        return None
    return str(val).strip().lower()

def clean_thai_name(val):
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip().lower()
    if val_str in ['none', 'null', 'nan', '']:
        return None
    # Remove all spaces
    val_str = re.sub(r'\s+', '', val_str)
    # Strip common prefixes
    prefixes = ['คุณ', 'นาย', 'นางสาว', 'นาง', 'น.ส.', 'เด็กชาย', 'เด็กหญิง', 'ด.ช.', 'ด.ญ.', 'ด.ช', 'ด.ญ']
    for p in prefixes:
        p_clean = p.lower()
        if val_str.startswith(p_clean):
            val_str = val_str[len(p_clean):]
            break
    return val_str if val_str else None

def check_coord_match(pred, gt):
    try:
        p_val = float(pred) if not pd.isna(pred) and pred is not None else 0.0
    except:
        p_val = 0.0
    try:
        g_val = float(gt) if not pd.isna(gt) and gt is not None else 0.0
    except:
        g_val = 0.0
        
    if p_val == 0.0 and g_val == 0.0:
        return 1
    return 1 if abs(p_val - g_val) < 0.001 else 0

def get_int_value(val):
    if pd.isna(val) or val is None:
        return 0
    try:
        return int(float(val))
    except:
        return 0

def main():
    results_dir = "e:/nlp-for-disaster/exp4/results"
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    comparison_metrics = []
    
    os.makedirs(os.path.join(results_dir, "confusion_matrices"), exist_ok=True)
    
    for model_name in models:
        csv_path = os.path.join(results_dir, f"{model_name}_results.csv")
        if not os.path.exists(csv_path):
            print(f"Result file for {model_name} not found at {csv_path}.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # 1. Stage 1 Evaluation (Informativeness / Help Request)
        # Convert true / pred values to booleans
        y_true_s1 = df['gt_is_help_request'].astype(bool)
        y_pred_s1 = df['pred_is_help_request'].fillna(False).astype(bool)
        
        s1_accuracy = accuracy_score(y_true_s1, y_pred_s1)
        s1_precision = precision_score(y_true_s1, y_pred_s1, zero_division=0)
        s1_recall = recall_score(y_true_s1, y_pred_s1, zero_division=0)
        s1_f1 = f1_score(y_true_s1, y_pred_s1, zero_division=0)
        
        # Draw Confusion Matrix for gt_is_help_request
        cm_s1 = confusion_matrix(y_true_s1, y_pred_s1)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_s1, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['False', 'True'], 
                    yticklabels=['False', 'True'])
        plt.title(f"Confusion Matrix (gt_is_help_request) - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrices/{model_name}_gt_is_help_request.png"))
        plt.close()

        # Draw Confusion Matrix for gt_classification_category
        # Map True -> help_request, False -> other
        y_true_cat = df['gt_classification_category'].fillna('other')
        y_pred_cat = df['pred_is_help_request'].fillna(False).apply(lambda x: 'help_request' if x else 'other')
        cm_cat = confusion_matrix(y_true_cat, y_pred_cat, labels=['other', 'help_request'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['other', 'help_request'], 
                    yticklabels=['other', 'help_request'])
        plt.title(f"Confusion Matrix (gt_classification_category) - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrices/{model_name}_gt_classification_category.png"))
        plt.close()
        
        # 2. Stage 2 Evaluation (Filtered by gt_is_help_request == True)
        df_stage2 = df[df['gt_is_help_request'] == True].copy()
        
        if df_stage2.empty:
            print(f"Warning: No rows with gt_is_help_request == True found for {model_name}!")
            continue

        # Draw Confusion Matrix for gt_victim_gender
        y_true_vg = df_stage2['gt_victim_gender'].apply(lambda x: clean_text_field(x) if pd.notna(x) else "None") if 'gt_victim_gender' in df_stage2.columns else pd.Series("None", index=df_stage2.index)
        pred_vg_col = 'pred_victim_gender' if 'pred_victim_gender' in df_stage2.columns else ('predicted_victim_gender' if 'predicted_victim_gender' in df_stage2.columns else None)
        y_pred_vg = df_stage2[pred_vg_col].apply(lambda x: clean_text_field(x) if pd.notna(x) else "None") if pred_vg_col else pd.Series("None", index=df_stage2.index)
        labels_gender = ['female', 'male', 'None']
        cm_vg = confusion_matrix(y_true_vg, y_pred_vg, labels=labels_gender)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_vg, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_gender, 
                    yticklabels=labels_gender)
        plt.title(f"Confusion Matrix (gt_victim_gender) - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrices/{model_name}_gt_victim_gender.png"))
        plt.close()

        # Draw Confusion Matrix for gt_reporter_gender
        y_true_rg = df_stage2['gt_reporter_gender'].apply(lambda x: clean_text_field(x) if pd.notna(x) else "None") if 'gt_reporter_gender' in df_stage2.columns else pd.Series("None", index=df_stage2.index)
        pred_rg_col = 'pred_reporter_gender' if 'pred_reporter_gender' in df_stage2.columns else ('predicted_reporter_gender' if 'predicted_reporter_gender' in df_stage2.columns else None)
        y_pred_rg = df_stage2[pred_rg_col].apply(lambda x: clean_text_field(x) if pd.notna(x) else "None") if pred_rg_col else pd.Series("None", index=df_stage2.index)
        cm_rg = confusion_matrix(y_true_rg, y_pred_rg, labels=labels_gender)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_rg, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_gender, 
                    yticklabels=labels_gender)
        plt.title(f"Confusion Matrix (gt_reporter_gender) - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrices/{model_name}_gt_reporter_gender.png"))
        plt.close()
            
        # Helper to get column with fallback
        def get_col_val(r, primary, secondary=None):
            if primary in r:
                return r[primary]
            if secondary and secondary in r:
                return r[secondary]
            # Try automatic mapping to predicted_ / gt_
            pred_fallback = primary.replace("pred_", "predicted_")
            if pred_fallback in r:
                return r[pred_fallback]
            return None

        # Text fields exact match rates
        victim_name_em = np.mean([1 if clean_thai_name(get_col_val(r, 'pred_victim_name')) == clean_thai_name(get_col_val(r, 'gt_victim_name')) else 0 for _, r in df_stage2.iterrows()])
        victim_nick_em = np.mean([1 if clean_thai_name(get_col_val(r, 'pred_victim_nickname')) == clean_thai_name(get_col_val(r, 'gt_victim_nickname')) else 0 for _, r in df_stage2.iterrows()])
        reporter_name_em = np.mean([1 if clean_thai_name(get_col_val(r, 'pred_reporter_name')) == clean_thai_name(get_col_val(r, 'gt_reporter_name')) else 0 for _, r in df_stage2.iterrows()])
        reporter_nick_em = np.mean([1 if clean_thai_name(get_col_val(r, 'pred_reporter_nickname')) == clean_thai_name(get_col_val(r, 'gt_reporter_nickname')) else 0 for _, r in df_stage2.iterrows()])
        location_em = np.mean([1 if clean_text_field(get_col_val(r, 'pred_location', 'predicted_location')) == clean_text_field(get_col_val(r, 'gt_location_name')) else 0 for _, r in df_stage2.iterrows()])
        map_url_em = np.mean([1 if clean_text_field(get_col_val(r, 'pred_google_map_url', 'predicted_google_map_url')) == clean_text_field(get_col_val(r, 'gt_google_map_url')) else 0 for _, r in df_stage2.iterrows()])
        
        # Phones EM
        victim_phone_em = np.mean([1 if clean_phone(get_col_val(r, 'pred_victim_phone')) == clean_phone(get_col_val(r, 'gt_victim_phone')) else 0 for _, r in df_stage2.iterrows()])
        reporter_phone_em = np.mean([1 if clean_phone(get_col_val(r, 'pred_reporter_phone')) == clean_phone(get_col_val(r, 'gt_reporter_phone')) else 0 for _, r in df_stage2.iterrows()])
        
        # Gender EM
        victim_gender_em = np.mean([1 if clean_text_field(get_col_val(r, 'pred_victim_gender')) == clean_text_field(get_col_val(r, 'gt_victim_gender')) else 0 for _, r in df_stage2.iterrows()])
        reporter_gender_em = np.mean([1 if clean_text_field(get_col_val(r, 'pred_reporter_gender')) == clean_text_field(get_col_val(r, 'gt_reporter_gender')) else 0 for _, r in df_stage2.iterrows()])
        
        # Coordinates EM
        lat_em = np.mean([check_coord_match(get_col_val(r, 'pred_lat'), get_col_val(r, 'gt_lat')) for _, r in df_stage2.iterrows()])
        lng_em = np.mean([check_coord_match(get_col_val(r, 'pred_lng'), get_col_val(r, 'gt_lng')) for _, r in df_stage2.iterrows()])
        
        # Numeric counts MAE & EM
        count_fields = [
            ('dead', 'pred_victims_dead', 'gt_dead'),
            ('critical', 'pred_victims_critical', 'gt_critical'),
            ('urgent', 'pred_victims_urgent', 'gt_urgent'),
            ('safe', 'pred_victims_safe', 'gt_safe'),
            ('child', 'pred_victims_child', 'gt_child'),
            ('bedridden', 'pred_victims_bedridden', 'gt_bedridden'),
            ('firstaid', 'pred_items_firstaid', 'gt_item_firstaid'),
            ('food', 'pred_items_food', 'gt_item_food'),
            ('energy', 'pred_items_energy', 'gt_item_energy'),
        ]
        
        count_metrics = {}
        for label, pred_col, gt_col in count_fields:
            # Safely check column in dataframe with fallback
            actual_pred_col = pred_col
            if pred_col not in df_stage2.columns:
                fallback_col = pred_col.replace("pred_", "predicted_")
                if fallback_col in df_stage2.columns:
                    actual_pred_col = fallback_col
                else:
                    actual_pred_col = None
            
            preds = df_stage2[actual_pred_col].apply(get_int_value) if actual_pred_col else pd.Series(0, index=df_stage2.index)
            gts = df_stage2[gt_col].apply(get_int_value) if gt_col in df_stage2.columns else pd.Series(0, index=df_stage2.index)
            
            mae = np.mean(np.abs(preds - gts))
            em = np.mean(preds == gts)
            f1_val = f1_score(gts, preds, average='weighted', zero_division=0)
            
            count_metrics[f"{label}_mae"] = mae
            count_metrics[f"{label}_em"] = em
            count_metrics[f"{label}_f1"] = f1_val
            
        # Collect all metrics
        row_metrics = {
            "model": model_name,
            "stage1_accuracy": s1_accuracy,
            "stage1_precision": s1_precision,
            "stage1_recall": s1_recall,
            "stage1_f1": s1_f1,
            "victim_name_em": victim_name_em,
            "victim_nickname_em": victim_nick_em,
            "victim_phone_em": victim_phone_em,
            "victim_gender_em": victim_gender_em,
            "reporter_name_em": reporter_name_em,
            "reporter_nickname_em": reporter_nick_em,
            "reporter_phone_em": reporter_phone_em,
            "reporter_gender_em": reporter_gender_em,
            "location_em": location_em,
            "google_map_url_em": map_url_em,
            "lat_em": lat_em,
            "lng_em": lng_em,
            "avg_latency_seconds": df['latency_seconds'].mean(),
            "total_tokens_used": int(df['token_in_use'].sum() + df['token_out_use'].sum())
        }
        
        # Merge count metrics
        row_metrics.update(count_metrics)
        comparison_metrics.append(row_metrics)
        
    metrics_df = pd.DataFrame(comparison_metrics)
    metrics_csv = os.path.join(results_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved Exp 4 comparison metrics to {metrics_csv}")
    
    # Print clean summary table
    summary_cols = ["model", "stage1_f1", "location_em", "victim_phone_em", "victim_name_em", "lat_em", "avg_latency_seconds"]
    print("\nSummary Table:")
    print(metrics_df[summary_cols].to_markdown(index=False))

if __name__ == '__main__':
    main()
