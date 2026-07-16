import os
import re
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Ensure relative imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import call_llm
from agent1_classification import classify_tweet
from agent2_ner import extract_ner
from agent3_1_people_extractor import extract_people
from agent3_2_pediatric_triage import triage_pediatric
from agent3_3_adult_triage import triage_adult

# Data paths
DATASET_PATH = "e:/nlp-for-disaster/dataset/clean/synthetic_ner_dataset.csv"
RESULTS_DIR = "e:/nlp-for-disaster/exp_full_agent/results"

# Price per million tokens in USD
PRICING = {
    "deepseek-v4-flash": {"input": 0.075, "output": 0.30},
    "gemma-4": {"input": 0.075, "output": 0.30},
    "typhoon-v2.5": {"input": 0.10, "output": 0.10}
}

# Default hetero configuration
HETERO_DEFAULT_CONFIG = {
    "agent1": "deepseek-v4-flash",
    "agent2": "gemma-4",
    "agent3_1": "typhoon-v2.5",
    "agent3_2": "typhoon-v2.5",
    "agent3_3": "typhoon-v2.5"
}

def clean_thai_name(val):
    if not val or str(val).strip().lower() in ['none', 'null', 'nan', '', 'ไม่ระบุชื่อ']:
        return None
    val_str = str(val).strip().lower()
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

def clean_text_field(val):
    if not val or str(val).strip().lower() in ['none', 'null', 'nan', '']:
        return None
    return str(val).strip().lower()

def clean_phone(phone_val):
    if not phone_val or str(phone_val).strip().lower() in ['none', 'null', 'nan', '']:
        return None
    return re.sub(r'\D', '', str(phone_val))

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

def parse_gt_victims(gt_victims_json_str):
    if not gt_victims_json_str or gt_victims_json_str == "[]" or pd.isna(gt_victims_json_str):
        return []
    try:
        return json.loads(gt_victims_json_str)
    except:
        return []

def calculate_cost(model, prompt_tokens, completion_tokens):
    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000.0

def align_people(gt_list, pred_list):
    """
    Aligns ground truth people list with predicted people list.
    Returns list of tuples: (gt_person, pred_person)
    """
    matched_gt = set()
    matched_pred = set()
    pairs = []
    
    # Clean and compare helper
    def get_name(p):
        return clean_thai_name(p.get("name")) if p else None
        
    # Match 1: Exact cleaned name match (excluding empty/unspecified names)
    for i, gt in enumerate(gt_list):
        gt_name = get_name(gt)
        if not gt_name:
            continue
        for j, pr in enumerate(pred_list):
            if j in matched_pred:
                continue
            pr_name = get_name(pr)
            if pr_name == gt_name:
                pairs.append((gt, pr))
                matched_gt.add(i)
                matched_pred.add(j)
                break
                
    # Match 2: Substring / Nickname name match
    for i, gt in enumerate(gt_list):
        if i in matched_gt:
            continue
        gt_name = get_name(gt)
        gt_nick = clean_thai_name(gt.get("nickname"))
        for j, pr in enumerate(pred_list):
            if j in matched_pred:
                continue
            pr_name = get_name(pr)
            pr_nick = clean_thai_name(pr.get("nickname"))
            
            if (gt_name and pr_name and (gt_name in pr_name or pr_name in gt_name)) or \
               (gt_nick and pr_name and gt_nick == pr_name) or \
               (gt_name and pr_nick and gt_name == pr_nick) or \
               (gt_nick and pr_nick and gt_nick == pr_nick):
                pairs.append((gt, pr))
                matched_gt.add(i)
                matched_pred.add(j)
                break
                
    # Match 3: Order-based match for remaining unmatched
    for i, gt in enumerate(gt_list):
        if i in matched_gt:
            continue
        for j, pr in enumerate(pred_list):
            if j in matched_pred:
                continue
            pairs.append((gt, pr))
            matched_gt.add(i)
            matched_pred.add(j)
            break
            
    # Add unmatched gt
    for i, gt in enumerate(gt_list):
        if i not in matched_gt:
            pairs.append((gt, None))
            
    # Add unmatched pred
    for j, pr in enumerate(pred_list):
        if j not in matched_pred:
            pairs.append((None, pr))
            
    return pairs

def bootstrap_ci(y_true, y_pred, metric_func, confidence=0.95, n_bootstraps=1000, **kwargs):
    """
    Computes confidence interval using bootstrapping.
    """
    if len(y_true) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            score = metric_func(y_true[indices], y_pred[indices], **kwargs)
            bootstrapped_scores.append(score)
        except Exception:
            pass
            
    if not bootstrapped_scores:
        return 0.0, 0.0
        
    sorted_scores = np.sort(bootstrapped_scores)
    alpha = 1.0 - confidence
    lower_idx = int(alpha / 2.0 * len(sorted_scores))
    upper_idx = int((1.0 - alpha / 2.0) * len(sorted_scores))
    lower_idx = max(0, min(lower_idx, len(sorted_scores) - 1))
    upper_idx = max(0, min(upper_idx, len(sorted_scores) - 1))
    
    return sorted_scores[lower_idx], sorted_scores[upper_idx]

def bootstrap_mean_ci(vals, confidence=0.95, n_bootstraps=1000):
    """
    Computes confidence interval for a mean using bootstrapping.
    """
    if not vals:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    scores = []
    vals_arr = np.array(vals)
    for _ in range(n_bootstraps):
        indices = rng.choice(len(vals_arr), size=len(vals_arr), replace=True)
        scores.append(np.mean(vals_arr[indices]))
    sorted_scores = np.sort(scores)
    alpha = 1.0 - confidence
    low_idx = int(alpha / 2.0 * len(sorted_scores))
    up_idx = int((1.0 - alpha / 2.0) * len(sorted_scores))
    low_idx = max(0, min(low_idx, len(sorted_scores) - 1))
    up_idx = max(0, min(up_idx, len(sorted_scores) - 1))
    return sorted_scores[low_idx], sorted_scores[up_idx]

def bootstrap_pipeline_metrics(merged_df, n_bootstraps=500, confidence=0.95):
    """
    Computes CI-95% for E2E metrics by bootstrapping the rows of the merged dataframe.
    """
    rng = np.random.default_rng(42)
    boot_metrics = {
        "classification_accuracy": [],
        "classification_f1": [],
        "e2e_triage_accuracy": [],
        "e2e_triage_f1": [],
        "location_em": [],
        "victim_phone_em": []
    }
    
    n_samples = len(merged_df)
    if n_samples == 0:
        return {k: (0.0, 0.0) for k in boot_metrics}
        
    for _ in range(n_bootstraps):
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_df = merged_df.iloc[boot_indices]
        
        # 1. Classification
        y_true_cls = boot_df["gt_is_help_request"].astype(bool)
        y_pred_cls = boot_df["pred_is_help_request"].astype(bool)
        cls_acc = accuracy_score(y_true_cls, y_pred_cls)
        cls_f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
        
        # 2. NER on true help requests
        df_help = boot_df[boot_df["gt_is_help_request"] == True]
        loc_em = 0.0
        v_phone_em = 0.0
        if not df_help.empty:
            loc_em = np.mean([1 if clean_text_field(r["pred_location_name"]) == clean_text_field(r["gt_location_name"]) else 0 for _, r in df_help.iterrows()])
            v_phone_em = np.mean([1 if clean_phone(r["pred_victim_phone"]) == clean_phone(r["gt_victim_phone"]) else 0 for _, r in df_help.iterrows()])
            
        # 3. E2E Triage
        total_people_gt = 0
        correct_triage = 0
        y_true_all = []
        y_pred_all = []
        for _, r in boot_df.iterrows():
            gt_list = parse_gt_victims(r["gt_victims_json"])
            pred_list = []
            if r["pred_is_help_request"]:
                try:
                    pred_list = json.loads(r["pred_victims_list"])
                except:
                    pass
            total_people_gt += len(gt_list)
            pairs = align_people(gt_list, pred_list)
            for gt_p, pr_p in pairs:
                gt_t = gt_p.get("triage_color") if gt_p else "NONE"
                pr_t = pr_p.get("triage_color") if pr_p else "NONE"
                y_true_all.append(gt_t)
                y_pred_all.append(pr_t)
                if gt_p and pr_p:
                    if gt_p.get("triage_color") == pr_p.get("triage_color"):
                        correct_triage += 1
        e2e_triage_acc = correct_triage / total_people_gt if total_people_gt > 0 else 1.0
        e2e_triage_f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0) if y_true_all else 1.0
        
        boot_metrics["classification_accuracy"].append(cls_acc)
        boot_metrics["classification_f1"].append(cls_f1)
        boot_metrics["e2e_triage_accuracy"].append(e2e_triage_acc)
        boot_metrics["e2e_triage_f1"].append(e2e_triage_f1)
        boot_metrics["location_em"].append(loc_em)
        boot_metrics["victim_phone_em"].append(v_phone_em)
        
    ci_results = {}
    alpha = 1.0 - confidence
    for k, scores in boot_metrics.items():
        if not scores:
            ci_results[k] = (0.0, 0.0)
            continue
        sorted_scores = np.sort(scores)
        low_idx = int(alpha / 2.0 * len(sorted_scores))
        up_idx = int((1.0 - alpha / 2.0) * len(sorted_scores))
        low_idx = max(0, min(low_idx, len(sorted_scores) - 1))
        up_idx = max(0, min(up_idx, len(sorted_scores) - 1))
        ci_results[k] = (sorted_scores[low_idx], sorted_scores[up_idx])
        
    return ci_results

def run_phase1_classification(df, models, max_workers):
    print("\n--- PHASE 1: Agent 1 (Classification) Evaluation ---")
    results = {}
    
    def process_item(row, model):
        pred, p_tok, c_tok, lat = classify_tweet(row['generated_text'], model)
        return {
            "synthetic_id": row['synthetic_id'],
            "pred_is_help": pred.get("is_help_request", False),
            "pred_category": pred.get("category", "other"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency": lat
        }
        
    for model in models:
        print(f"Evaluating {model}...")
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent1_classification_{model}.csv")
        existing_data = {}
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty and 'synthetic_id' in existing_df.columns:
                    for _, r in existing_df.iterrows():
                        latency_val = float(r.get('latency', 0.0))
                        prompt_toks = int(r.get('prompt_tokens', 0))
                        if latency_val == 0.0 or prompt_toks == 0:
                            continue
                        existing_data[r['synthetic_id']] = {
                            "synthetic_id": r['synthetic_id'],
                            "pred_is_help": bool(r.get('pred_is_help')),
                            "pred_category": r.get('pred_category'),
                            "prompt_tokens": prompt_toks,
                            "completion_tokens": int(r.get('completion_tokens', 0)),
                            "latency": latency_val
                        }
                    print(f"Loaded {len(existing_data)} existing classification results for {model} from {csv_path}.")
            except Exception as e:
                print(f"Error loading existing cache for {model}: {e}")
                
        # Find which rows need to be processed
        rows_to_process = [row for _, row in df.iterrows() if row['synthetic_id'] not in existing_data]
        outputs = [existing_data[row['synthetic_id']] for _, row in df.iterrows() if row['synthetic_id'] in existing_data]
        
        if rows_to_process:
            print(f"Processing {len(rows_to_process)} remaining rows for {model}...")
            new_outputs = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_item, row, model): row for row in rows_to_process}
                for f in as_completed(futures):
                    new_outputs.append(f.result())
            outputs.extend(new_outputs)
            
        out_df = pd.DataFrame(outputs)
        merged = df.merge(out_df, on="synthetic_id")
        
        # Calculate metrics
        y_true = merged['gt_is_help_request'].astype(bool)
        y_pred = merged['pred_is_help'].astype(bool)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
        f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, zero_division=0)
        
        avg_lat = merged['latency'].mean()
        tot_p = merged['prompt_tokens'].sum()
        tot_c = merged['completion_tokens'].sum()
        tot_cost = calculate_cost(model, tot_p, tot_c)
        
        results[model] = {
            "accuracy": acc,
            "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
            "avg_latency": avg_lat,
            "total_input_tokens": int(tot_p),
            "total_output_tokens": int(tot_c),
            "cost": tot_cost
        }
        
        # Save individual model classification results
        os.makedirs(os.path.join(RESULTS_DIR, "phase_1"), exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent1_classification_{model}.csv")
        merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved Classification details for {model} to: {csv_path}")
        
    res_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={"index": "model"})
    comp_csv = os.path.join(RESULTS_DIR, "phase_1/agent1_classification_comparison.csv")
    res_df.to_csv(comp_csv, index=False)
    print(f"Saved Classification comparison to: {comp_csv}")
    print("Classification comparison:")
    print(res_df.to_markdown(index=False))
    return results

def run_phase1_ner(df, models, max_workers):
    print("\n--- PHASE 1: Agent 2 (NER) Evaluation ---")
    df_help = df[df['gt_is_help_request'] == True].copy()
    if df_help.empty:
        print("No help requests found in dataset to evaluate NER.")
        return {}
        
    results = {}
    
    def process_item(row, model):
        pred, p_tok, c_tok, lat = extract_ner(row['generated_text'], model)
        return {
            "synthetic_id": row['synthetic_id'],
            "pred": pred,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency": lat
        }
        
    import ast
    for model in models:
        print(f"Evaluating {model}...")
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent2_ner_{model}.csv")
        existing_data = {}
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty and 'synthetic_id' in existing_df.columns:
                    for _, r in existing_df.iterrows():
                        latency_val = float(r.get('latency', 0.0))
                        prompt_toks = int(r.get('prompt_tokens', 0))
                        if latency_val == 0.0 or prompt_toks == 0:
                            continue
                        pred_val = r.get('pred')
                        parsed_pred = {}
                        if pd.notna(pred_val) and str(pred_val).strip() != "":
                            try:
                                parsed_pred = json.loads(str(pred_val))
                            except Exception:
                                try:
                                    parsed_pred = ast.literal_eval(str(pred_val))
                                except Exception:
                                    pass
                        existing_data[r['synthetic_id']] = {
                            "synthetic_id": r['synthetic_id'],
                            "pred": parsed_pred,
                            "prompt_tokens": prompt_toks,
                            "completion_tokens": int(r.get('completion_tokens', 0)),
                            "latency": latency_val
                        }
                    print(f"Loaded {len(existing_data)} existing NER results for {model} from {csv_path}.")
            except Exception as e:
                print(f"Error loading existing cache for {model}: {e}")
                
        # Find which rows need to be processed
        rows_to_process = [row for _, row in df_help.iterrows() if row['synthetic_id'] not in existing_data]
        outputs = [existing_data[row['synthetic_id']] for _, row in df_help.iterrows() if row['synthetic_id'] in existing_data]
        
        if rows_to_process:
            print(f"Processing {len(rows_to_process)} remaining rows for {model}...")
            new_outputs = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_item, row, model): row for row in rows_to_process}
                for f in as_completed(futures):
                    new_outputs.append(f.result())
            outputs.extend(new_outputs)
            
        out_df = pd.DataFrame(outputs)
        merged = df_help.merge(out_df, on="synthetic_id")
        
        # Compute exact match scores for each row
        location_matches = []
        map_url_matches = []
        lat_matches = []
        lng_matches = []
        victim_name_matches = []
        victim_phone_matches = []
        reporter_name_matches = []
        reporter_phone_matches = []
        
        dead_ems = []
        critical_ems = []
        urgent_ems = []
        safe_ems = []
        child_ems = []
        bedridden_ems = []
        
        firstaid_ems = []
        food_ems = []
        energy_ems = []
        
        for _, r in merged.iterrows():
            pred = r['pred'] or {}
            
            # Location
            pred_coords = pred.get("coordinates", {})
            location_matches.append(1 if clean_text_field(pred_coords.get("name")) == clean_text_field(r['gt_location_name']) else 0)
            map_url_matches.append(1 if clean_text_field(pred_coords.get("google_map_url")) == clean_text_field(r['gt_google_map_url']) else 0)
            lat_matches.append(check_coord_match(pred_coords.get("lat"), r['gt_lat']))
            lng_matches.append(check_coord_match(pred_coords.get("lng"), r['gt_lng']))
            
            # Contacts
            cv = pred.get("contact_victim", [])
            pred_v_name = cv[0].get("name") if (isinstance(cv, list) and len(cv) > 0) else None
            pred_v_phone = cv[0].get("phone") if (isinstance(cv, list) and len(cv) > 0) else None
            victim_name_matches.append(1 if clean_thai_name(pred_v_name) == clean_thai_name(r['gt_victim_name']) else 0)
            victim_phone_matches.append(1 if clean_phone(pred_v_phone) == clean_phone(r['gt_victim_phone']) else 0)
            
            cr = pred.get("contact_reporter", [])
            pred_r_name = cr[0].get("name") if (isinstance(cr, list) and len(cr) > 0) else None
            pred_r_phone = cr[0].get("phone") if (isinstance(cr, list) and len(cr) > 0) else None
            reporter_name_matches.append(1 if clean_thai_name(pred_r_name) == clean_thai_name(r['gt_reporter_name']) else 0)
            reporter_phone_matches.append(1 if clean_phone(pred_r_phone) == clean_phone(r['gt_reporter_phone']) else 0)
            
            # Victims count
            v_counts = pred.get("victims", {})
            dead_ems.append(1 if get_int_value(v_counts.get("dead")) == get_int_value(r['gt_dead']) else 0)
            critical_ems.append(1 if get_int_value(v_counts.get("critical")) == get_int_value(r['gt_critical']) else 0)
            urgent_ems.append(1 if get_int_value(v_counts.get("urgent")) == get_int_value(r['gt_urgent']) else 0)
            safe_ems.append(1 if get_int_value(v_counts.get("safe")) == get_int_value(r['gt_safe']) else 0)
            child_ems.append(1 if get_int_value(v_counts.get("child")) == get_int_value(r['gt_child']) else 0)
            bedridden_ems.append(1 if get_int_value(v_counts.get("bedridden")) == get_int_value(r['gt_bedridden']) else 0)
            
            # Items count
            i_counts = pred.get("items", {})
            firstaid_ems.append(1 if get_int_value(i_counts.get("firstAid")) == get_int_value(r['gt_item_firstaid']) else 0)
            food_ems.append(1 if get_int_value(i_counts.get("food")) == get_int_value(r['gt_item_food']) else 0)
            energy_ems.append(1 if get_int_value(i_counts.get("energy")) == get_int_value(r['gt_item_energy']) else 0)
            
        merged['location_em'] = location_matches
        merged['map_url_em'] = map_url_matches
        merged['lat_em'] = lat_matches
        merged['lng_em'] = lng_matches
        merged['victim_name_em'] = victim_name_matches
        merged['victim_phone_em'] = victim_phone_matches
        merged['reporter_name_em'] = reporter_name_matches
        merged['reporter_phone_em'] = reporter_phone_matches
        
        merged['dead_em'] = dead_ems
        merged['critical_em'] = critical_ems
        merged['urgent_em'] = urgent_ems
        
        avg_lat = merged['latency'].mean()
        tot_p = merged['prompt_tokens'].sum()
        tot_c = merged['completion_tokens'].sum()
        tot_cost = calculate_cost(model, tot_p, tot_c)
        
        # Calculate field averages
        loc_ci = bootstrap_mean_ci(location_matches)
        v_phone_ci = bootstrap_mean_ci(victim_phone_matches)
        crit_ci = bootstrap_mean_ci(critical_ems)
        
        results[model] = {
            "location_em": np.mean(location_matches),
            "location_em_ci": f"{loc_ci[0]:.3f} - {loc_ci[1]:.3f}",
            "map_url_em": np.mean(map_url_matches),
            "lat_em": np.mean(lat_matches),
            "lng_em": np.mean(lng_matches),
            "victim_name_em": np.mean(victim_name_matches),
            "victim_phone_em": np.mean(victim_phone_matches),
            "victim_phone_em_ci": f"{v_phone_ci[0]:.3f} - {v_phone_ci[1]:.3f}",
            "reporter_name_em": np.mean(reporter_name_matches),
            "reporter_phone_em": np.mean(reporter_phone_matches),
            "avg_latency": avg_lat,
            "total_input_tokens": int(tot_p),
            "total_output_tokens": int(tot_c),
            "cost": tot_cost,
            "dead_em": np.mean(dead_ems),
            "critical_em": np.mean(critical_ems),
            "critical_em_ci": f"{crit_ci[0]:.3f} - {crit_ci[1]:.3f}",
            "urgent_em": np.mean(urgent_ems),
            "safe_em": np.mean(safe_ems),
            "child_em": np.mean(child_ems),
            "bedridden_em": np.mean(bedridden_ems),
            "firstaid_em": np.mean(firstaid_ems),
            "food_em": np.mean(food_ems),
            "energy_em": np.mean(energy_ems)
        }
        
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent2_ner_{model}.csv")
        merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved NER details for {model} to: {csv_path}")
        
    res_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={"index": "model"})
    comp_csv = os.path.join(RESULTS_DIR, "phase_1/agent2_ner_comparison.csv")
    res_df.to_csv(comp_csv, index=False)
    print(f"Saved NER comparison to: {comp_csv}")
    
    print("NER comparison (selected fields):")
    print(res_df[["model", "location_em", "victim_name_em", "victim_phone_em", "critical_em", "avg_latency"]].to_markdown(index=False))
    return results

def run_phase1_people_extractor(df, models, max_workers):
    print("\n--- PHASE 1: Agent 3.1 (People Extractor) Evaluation ---")
    df_help = df[df['gt_is_help_request'] == True].copy()
    if df_help.empty:
        print("No help requests found in dataset to evaluate People Extractor.")
        return {}
        
    results = {}
    
    def process_item(row, model):
        pred, p_tok, c_tok, lat = extract_people(row['generated_text'], model)
        return {
            "synthetic_id": row['synthetic_id'],
            "pred_people": pred.get("people", []),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency": lat
        }
        
    import ast
    for model in models:
        print(f"Evaluating {model}...")
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_1_people_extractor_{model}.csv")
        existing_data = {}
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty and 'synthetic_id' in existing_df.columns:
                    for _, r in existing_df.iterrows():
                        latency_val = float(r.get('latency', 0.0))
                        prompt_toks = int(r.get('prompt_tokens', 0))
                        if latency_val == 0.0 or prompt_toks == 0:
                            continue
                        people_val = r.get('pred_people')
                        parsed_people = []
                        if pd.notna(people_val) and str(people_val).strip() != "":
                            try:
                                parsed_people = json.loads(str(people_val))
                            except Exception:
                                try:
                                    parsed_people = ast.literal_eval(str(people_val))
                                except Exception:
                                    pass
                        existing_data[r['synthetic_id']] = {
                            "synthetic_id": r['synthetic_id'],
                            "pred_people": parsed_people,
                            "prompt_tokens": prompt_toks,
                            "completion_tokens": int(r.get('completion_tokens', 0)),
                            "latency": latency_val
                        }
                    print(f"Loaded {len(existing_data)} existing People Extractor results for {model} from {csv_path}.")
            except Exception as e:
                print(f"Error loading existing cache for {model}: {e}")
                
        # Find which rows need to be processed
        rows_to_process = [row for _, row in df_help.iterrows() if row['synthetic_id'] not in existing_data]
        outputs = [existing_data[row['synthetic_id']] for _, row in df_help.iterrows() if row['synthetic_id'] in existing_data]
        
        if rows_to_process:
            print(f"Processing {len(rows_to_process)} remaining rows for {model}...")
            new_outputs = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_item, row, model): row for row in rows_to_process}
                for f in as_completed(futures):
                    new_outputs.append(f.result())
            outputs.extend(new_outputs)
            
        out_df = pd.DataFrame(outputs)
        merged = df_help.merge(out_df, on="synthetic_id")
        
        exact_counts = []
        age_group_accs = []
        
        for _, r in merged.iterrows():
            gt_list = parse_gt_victims(r['gt_victims_json'])
            pred_list = r['pred_people']
            
            # 1. Exact count match
            exact_counts.append(1 if len(gt_list) == len(pred_list) else 0)
            
            # 2. Age group accuracy
            pairs = align_people(gt_list, pred_list)
            correct_age_groups = 0
            for gt_p, pr_p in pairs:
                if gt_p and pr_p:
                    gt_ag = gt_p.get("age_group")
                    pr_ag = pr_p.get("age_group")
                    if gt_ag == pr_ag:
                        correct_age_groups += 1
                        
            total_gt = len(gt_list)
            age_group_accs.append(correct_age_groups / total_gt if total_gt > 0 else 1.0)
            
        merged['exact_count_match'] = exact_counts
        merged['age_group_acc'] = age_group_accs
        
        avg_lat = merged['latency'].mean()
        tot_p = merged['prompt_tokens'].sum()
        tot_c = merged['completion_tokens'].sum()
        tot_cost = calculate_cost(model, tot_p, tot_c)
        
        count_ci = bootstrap_mean_ci(exact_counts)
        age_ci = bootstrap_mean_ci(age_group_accs)
        
        results[model] = {
            "exact_count_match_rate": np.mean(exact_counts),
            "exact_count_match_rate_ci": f"{count_ci[0]:.3f} - {count_ci[1]:.3f}",
            "age_group_accuracy": np.mean(age_group_accs),
            "age_group_accuracy_ci": f"{age_ci[0]:.3f} - {age_ci[1]:.3f}",
            "avg_latency": avg_lat,
            "total_input_tokens": int(tot_p),
            "total_output_tokens": int(tot_c),
            "cost": tot_cost
        }
        
        csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_1_people_extractor_{model}.csv")
        merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved People Extractor details for {model} to: {csv_path}")
        
    res_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={"index": "model"})
    comp_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_1_people_extractor_comparison.csv")
    res_df.to_csv(comp_csv, index=False)
    print(f"Saved People Extractor comparison to: {comp_csv}")
    print("People Extractor comparison:")
    print(res_df.to_markdown(index=False))
    return results

def run_phase1_triage(df, models, max_workers):
    print("\n--- PHASE 1: Agent 3.2 & 3.3 (Triage) Evaluation ---")
    df_help = df[df['gt_is_help_request'] == True].copy()
    if df_help.empty:
        print("No help requests found in dataset to evaluate Triage.")
        return {}, {}
        
    # Extract all individual victims to run independently
    pediatric_victims = []
    adult_victims = []
    
    for _, row in df_help.iterrows():
        gt_list = parse_gt_victims(row['gt_victims_json'])
        for v in gt_list:
            v_info = {
                "name": v.get("name") or "ไม่ระบุชื่อ",
                "age": v.get("age"),
                "age_group": v.get("age_group"),
                "symptoms_literal": v.get("symptoms_literal"),
                "gt_triage": v.get("triage_color")
            }
            if v_info["age_group"] == "child" or (v_info["age"] is not None and v_info["age"] < 12):
                pediatric_victims.append(v_info)
            else:
                adult_victims.append(v_info)
                
    print(f"Extracted {len(pediatric_victims)} pediatric ground truth victims and {len(adult_victims)} adult ground truth victims.")
    
    ped_results = {}
    ad_results = {}
    
    # 1. Evaluate Pediatric Triage
    if pediatric_victims:
        def process_pediatric(v, model):
            age_val = v['age'] if v['age'] is not None else 11
            pred, p_tok, c_tok, lat = triage_pediatric(v['name'], age_val, v['symptoms_literal'], model)
            return {
                "name": v['name'],
                "age": age_val,
                "symptoms_literal": v['symptoms_literal'],
                "gt_triage": v['gt_triage'],
                "pred_triage": pred.get("triage_color"),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "latency": lat
            }
            
        for model in models:
            print(f"Evaluating Pediatric Triage ({model})...")
            csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_2_pediatric_triage_{model}.csv")
            existing_data = {}
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if not existing_df.empty and 'name' in existing_df.columns and 'symptoms_literal' in existing_df.columns:
                        for _, r in existing_df.iterrows():
                            k = (str(r.get('name')).strip(), str(r.get('symptoms_literal')).strip())
                            existing_data[k] = {
                                "name": r.get('name'),
                                "age": r.get('age'),
                                "symptoms_literal": r.get('symptoms_literal'),
                                "gt_triage": r.get('gt_triage'),
                                "pred_triage": r.get('pred_triage'),
                                "prompt_tokens": int(r.get('prompt_tokens', 0)),
                                "completion_tokens": int(r.get('completion_tokens', 0)),
                                "latency": float(r.get('latency', 0.0))
                            }
                        print(f"Loaded {len(existing_data)} existing Pediatric Triage results for {model} from cache.")
                except Exception as e:
                    print(f"Error loading existing cache for Pediatric Triage ({model}): {e}")
            
            # Filter victims to process
            victims_to_process = []
            outputs = []
            for v in pediatric_victims:
                k = (str(v['name']).strip(), str(v['symptoms_literal']).strip())
                if k in existing_data:
                    outputs.append(existing_data[k])
                else:
                    victims_to_process.append(v)
            
            if victims_to_process:
                print(f"Processing {len(victims_to_process)} remaining pediatric victims for {model}...")
                new_outputs = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_pediatric, v, model): v for v in victims_to_process}
                    for f in as_completed(futures):
                        new_outputs.append(f.result())
                outputs.extend(new_outputs)
                    
            out_df = pd.DataFrame(outputs)
            y_true = out_df['gt_triage'].fillna("GREEN")
            y_pred = out_df['pred_triage'].fillna("GREEN")
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
            f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, average='weighted', zero_division=0)
            
            avg_lat = out_df['latency'].mean()
            tot_p = out_df['prompt_tokens'].sum()
            tot_c = out_df['completion_tokens'].sum()
            tot_cost = calculate_cost(model, tot_p, tot_c)
            
            ped_results[model] = {
                "accuracy": acc,
                "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
                "f1_score": f1,
                "f1_score_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
                "avg_latency": avg_lat,
                "total_input_tokens": int(tot_p),
                "total_output_tokens": int(tot_c),
                "cost": tot_cost
            }
            csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_2_pediatric_triage_{model}.csv")
            out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved Pediatric Triage details for {model} to: {csv_path}")
            
        ped_df = pd.DataFrame.from_dict(ped_results, orient='index').reset_index().rename(columns={"index": "model"})
        comp_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_2_pediatric_triage_comparison.csv")
        ped_df.to_csv(comp_csv, index=False)
        print(f"Saved Pediatric Triage comparison to: {comp_csv}")
        print("Pediatric Triage comparison:")
        print(ped_df.to_markdown(index=False))
        
    # 2. Evaluate Adult Triage
    if adult_victims:
        def process_adult(v, model):
            pred, p_tok, c_tok, lat = triage_adult(v['name'], v['age'], v['symptoms_literal'], model)
            return {
                "name": v['name'],
                "age": v['age'],
                "symptoms_literal": v['symptoms_literal'],
                "gt_triage": v['gt_triage'],
                "pred_triage": pred.get("triage_color"),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "latency": lat
            }
            
        for model in models:
            print(f"Evaluating Adult Triage ({model})...")
            csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_3_adult_triage_{model}.csv")
            existing_data = {}
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if not existing_df.empty and 'name' in existing_df.columns and 'symptoms_literal' in existing_df.columns:
                        for _, r in existing_df.iterrows():
                            k = (str(r.get('name')).strip(), str(r.get('symptoms_literal')).strip())
                            existing_data[k] = {
                                "name": r.get('name'),
                                "age": r.get('age'),
                                "symptoms_literal": r.get('symptoms_literal'),
                                "gt_triage": r.get('gt_triage'),
                                "pred_triage": r.get('pred_triage'),
                                "prompt_tokens": int(r.get('prompt_tokens', 0)),
                                "completion_tokens": int(r.get('completion_tokens', 0)),
                                "latency": float(r.get('latency', 0.0))
                            }
                        print(f"Loaded {len(existing_data)} existing Adult Triage results for {model} from cache.")
                except Exception as e:
                    print(f"Error loading existing cache for Adult Triage ({model}): {e}")
            
            # Filter victims to process
            victims_to_process = []
            outputs = []
            for v in adult_victims:
                k = (str(v['name']).strip(), str(v['symptoms_literal']).strip())
                if k in existing_data:
                    outputs.append(existing_data[k])
                else:
                    victims_to_process.append(v)
            
            if victims_to_process:
                print(f"Processing {len(victims_to_process)} remaining adult victims for {model}...")
                new_outputs = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_adult, v, model): v for v in victims_to_process}
                    for f in as_completed(futures):
                        new_outputs.append(f.result())
                outputs.extend(new_outputs)
                    
            out_df = pd.DataFrame(outputs)
            y_true = out_df['gt_triage'].fillna("GREEN")
            y_pred = out_df['pred_triage'].fillna("GREEN")
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
            f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, average='weighted', zero_division=0)
            
            avg_lat = out_df['latency'].mean()
            tot_p = out_df['prompt_tokens'].sum()
            tot_c = out_df['completion_tokens'].sum()
            tot_cost = calculate_cost(model, tot_p, tot_c)
            
            ad_results[model] = {
                "accuracy": acc,
                "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
                "f1_score": f1,
                "f1_score_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
                "avg_latency": avg_lat,
                "total_input_tokens": int(tot_p),
                "total_output_tokens": int(tot_c),
                "cost": tot_cost
            }
            csv_path = os.path.join(RESULTS_DIR, f"phase_1/agent3_3_adult_triage_{model}.csv")
            out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved Adult Triage details for {model} to: {csv_path}")
            
        ad_df = pd.DataFrame.from_dict(ad_results, orient='index').reset_index().rename(columns={"index": "model"})
        comp_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_3_adult_triage_comparison.csv")
        ad_df.to_csv(comp_csv, index=False)
        print(f"Saved Adult Triage comparison to: {comp_csv}")
        print("Adult Triage comparison:")
        print(ad_df.to_markdown(index=False))
        
    return ped_results, ad_results

def run_pipeline_e2e(df, config, config_name, max_workers, phase_num):
    """
    Runs the full pipeline end-to-end for a given model config.
    config: dict mapping 'agent1', 'agent2', 'agent3_1', 'agent3_2', 'agent3_3' to model ids
    """
    print(f"\nRunning Full Pipeline E2E ({config_name}) for Phase {phase_num}...")
    
    # 1. Define paths
    phase_dir = os.path.join(RESULTS_DIR, f"phase_{phase_num}")
    os.makedirs(phase_dir, exist_ok=True)
    
    pipeline_csv = os.path.join(phase_dir, f"{config_name}_full_pipeline.csv")
    a1_csv = os.path.join(phase_dir, f"{config_name}_agent1_classification.csv")
    a2_csv = os.path.join(phase_dir, f"{config_name}_agent2_ner.csv")
    a3_1_csv = os.path.join(phase_dir, f"{config_name}_agent3_1_people_extractor.csv")
    a3_2_csv = os.path.join(phase_dir, f"{config_name}_agent3_2_pediatric_triage.csv")
    a3_3_csv = os.path.join(phase_dir, f"{config_name}_agent3_3_adult_triage.csv")
    
    # Cost calculation helper (needs config from run_pipeline_e2e scope)
    def calc_row_cost(r):
        cost = 0.0
        if all(config[k] == config["agent1"] for k in config):
            cost = calculate_cost(config["agent1"], r["prompt_tokens"], r["completion_tokens"])
        else:
            cost = 0.0
            cost += calculate_cost(config["agent1"], r["prompt_tokens"] * 0.2, r["completion_tokens"] * 0.1)
            if r["pred_is_help_request"]:
                cost += calculate_cost(config["agent2"], r["prompt_tokens"] * 0.3, r["completion_tokens"] * 0.3)
                cost += calculate_cost(config["agent3_1"], r["prompt_tokens"] * 0.3, r["completion_tokens"] * 0.3)
                cost += calculate_cost(config["agent3_2"], r["prompt_tokens"] * 0.2, r["completion_tokens"] * 0.3)
        return cost
        
    # 2. Check cache / Load existing files
    existing_ids = set()
    existing_pipeline_df = pd.DataFrame()
    if os.path.exists(pipeline_csv):
        try:
            existing_pipeline_df = pd.read_csv(pipeline_csv)
            if not existing_pipeline_df.empty and 'synthetic_id' in existing_pipeline_df.columns:
                existing_ids = set(existing_pipeline_df['synthetic_id'].dropna().tolist())
                print(f"Found {len(existing_ids)} existing processed rows for {config_name}.")
        except Exception as e:
            print(f"Error loading existing pipeline CSV: {e}")
            
    existing_a1_df = pd.read_csv(a1_csv) if os.path.exists(a1_csv) else pd.DataFrame()
    if not existing_a1_df.empty and 'synthetic_id' in existing_a1_df.columns:
        existing_a1_df = existing_a1_df[existing_a1_df['synthetic_id'].isin(existing_ids)]
        
    existing_a2_df = pd.read_csv(a2_csv) if os.path.exists(a2_csv) else pd.DataFrame()
    if not existing_a2_df.empty and 'synthetic_id' in existing_a2_df.columns:
        existing_a2_df = existing_a2_df[existing_a2_df['synthetic_id'].isin(existing_ids)]
        
    existing_a3_1_df = pd.read_csv(a3_1_csv) if os.path.exists(a3_1_csv) else pd.DataFrame()
    if not existing_a3_1_df.empty and 'synthetic_id' in existing_a3_1_df.columns:
        existing_a3_1_df = existing_a3_1_df[existing_a3_1_df['synthetic_id'].isin(existing_ids)]
        
    existing_a3_2_df = pd.read_csv(a3_2_csv) if os.path.exists(a3_2_csv) else pd.DataFrame()
    if not existing_a3_2_df.empty and 'synthetic_id' in existing_a3_2_df.columns:
        existing_a3_2_df = existing_a3_2_df[existing_a3_2_df['synthetic_id'].isin(existing_ids)]
        
    existing_a3_3_df = pd.read_csv(a3_3_csv) if os.path.exists(a3_3_csv) else pd.DataFrame()
    if not existing_a3_3_df.empty and 'synthetic_id' in existing_a3_3_df.columns:
        existing_a3_3_df = existing_a3_3_df[existing_a3_3_df['synthetic_id'].isin(existing_ids)]
        
    # Filter rows to process
    rows_to_process = [row for _, row in df.iterrows() if row['synthetic_id'] not in existing_ids]
    
    def run_row(row):
        start_time = time.time()
        
        # Log metrics per row
        total_p_tokens = 0
        total_c_tokens = 0
        stages_latencies = {}
        
        # Stage 1: Classification
        m1 = config["agent1"]
        pred1, p1, c1, lat1 = classify_tweet(row['generated_text'], m1)
        total_p_tokens += p1
        total_c_tokens += c1
        stages_latencies["classification"] = lat1
        
        is_help = pred1.get("is_help_request", False)
        
        pipeline_output = {
            "synthetic_id": row["synthetic_id"],
            "pred_is_help_request": is_help,
            "pred_location_name": None,
            "pred_google_map_url": None,
            "pred_lat": 0.0,
            "pred_lng": 0.0,
            "pred_victim_name": None,
            "pred_victim_phone": None,
            "pred_victims_list": "[]",
            "stages_latency_seconds": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_seconds": 0.0
        }
        
        a1_log = {
            "synthetic_id": row["synthetic_id"],
            "text": row["generated_text"],
            "gt_is_help_request": row["gt_is_help_request"],
            "pred_is_help_request": is_help,
            "pred_category": pred1.get("category", "other"),
            "latency": lat1,
            "prompt_tokens": p1,
            "completion_tokens": c1
        }
        
        a2_log = None
        a3_1_log = None
        a3_2_logs = []
        a3_3_logs = []
        
        if is_help:
            # Stage 2: NER
            m2 = config["agent2"]
            pred2, p2, c2, lat2 = extract_ner(row['generated_text'], m2)
            total_p_tokens += p2
            total_c_tokens += c2
            stages_latencies["ner"] = lat2
            
            coords = pred2.get("coordinates", {})
            pipeline_output["pred_location_name"] = coords.get("name")
            pipeline_output["pred_google_map_url"] = coords.get("google_map_url")
            pipeline_output["pred_lat"] = coords.get("lat", 0.0)
            pipeline_output["pred_lng"] = coords.get("lng", 0.0)
            
            cv = pred2.get("contact_victim", [])
            if isinstance(cv, list) and len(cv) > 0:
                pipeline_output["pred_victim_name"] = cv[0].get("name")
                pipeline_output["pred_victim_phone"] = cv[0].get("phone")
                
            a2_log = {
                "synthetic_id": row["synthetic_id"],
                "text": row["generated_text"],
                "gt_location_name": row["gt_location_name"],
                "pred_location_name": pipeline_output["pred_location_name"],
                "gt_google_map_url": row["gt_google_map_url"],
                "pred_google_map_url": pipeline_output["pred_google_map_url"],
                "gt_lat": row["gt_lat"],
                "pred_lat": pipeline_output["pred_lat"],
                "gt_lng": row["gt_lng"],
                "pred_lng": pipeline_output["pred_lng"],
                "gt_victim_name": row["gt_victim_name"],
                "pred_victim_name": pipeline_output["pred_victim_name"],
                "gt_victim_phone": row["gt_victim_phone"],
                "pred_victim_phone": pipeline_output["pred_victim_phone"],
                "latency": lat2,
                "prompt_tokens": p2,
                "completion_tokens": c2
            }
            
            # Stage 3.1: People Extractor
            m3_1 = config["agent3_1"]
            pred3_1, p3_1, c3_1, lat3_1 = extract_people(row['generated_text'], m3_1)
            total_p_tokens += p3_1
            total_c_tokens += c3_1
            stages_latencies["people_extractor"] = lat3_1
            
            people = pred3_1.get("people", [])
            stages_latencies["triage"] = 0.0
            
            a3_1_log = {
                "synthetic_id": row["synthetic_id"],
                "text": row["generated_text"],
                "gt_victims_json": row["gt_victims_json"],
                "pred_people_json": json.dumps(people, ensure_ascii=False),
                "latency": lat3_1,
                "prompt_tokens": p3_1,
                "completion_tokens": c3_1
            }
            
            # Triage each extracted person
            gt_list = parse_gt_victims(row["gt_victims_json"])
            pairs = align_people(gt_list, people)
            pred_to_gt = {}
            for gt_p, pr_p in pairs:
                if pr_p and gt_p:
                    pred_to_gt[id(pr_p)] = gt_p
            
            triaged_people = []
            for p in people:
                name = p.get("name") or "ไม่ระบุชื่อ"
                age = p.get("age")
                age_group = p.get("age_group")
                symptoms = p.get("symptoms_literal") or ""
                
                is_child = age_group == "child" or (age is not None and age < 12)
                
                if is_child:
                    m3_2 = config["agent3_2"]
                    pred_t, pt, ct, latt = triage_pediatric(name, age or 11, symptoms, m3_2)
                else:
                    m3_3 = config["agent3_3"]
                    pred_t, pt, ct, latt = triage_adult(name, age, symptoms, m3_3)
                    
                total_p_tokens += pt
                total_c_tokens += ct
                stages_latencies["triage"] += latt
                
                p["triage_color"] = pred_t.get("triage_color", "GREEN")
                p["triage_reasoning"] = pred_t.get("reasoning", "")
                triaged_people.append(p)
                
                gt_p = pred_to_gt.get(id(p))
                gt_triage = gt_p.get("triage_color") if gt_p else None
                
                triage_log = {
                    "synthetic_id": row["synthetic_id"],
                    "victim_name": name,
                    "victim_age": age,
                    "symptoms_literal": symptoms,
                    "gt_triage": gt_triage,
                    "pred_triage": p["triage_color"],
                    "reasoning": p["triage_reasoning"],
                    "latency": latt,
                    "prompt_tokens": pt,
                    "completion_tokens": ct
                }
                
                if is_child:
                    a3_2_logs.append(triage_log)
                else:
                    a3_3_logs.append(triage_log)
                    
            pipeline_output["pred_victims_list"] = json.dumps(triaged_people, ensure_ascii=False)
            
        pipeline_output["prompt_tokens"] = total_p_tokens
        pipeline_output["completion_tokens"] = total_c_tokens
        pipeline_output["latency_seconds"] = time.time() - start_time
        pipeline_output["stages_latency_seconds"] = json.dumps(stages_latencies)
        
        return {
            "pipeline_output": pipeline_output,
            "agent1_log": a1_log,
            "agent2_log": a2_log,
            "agent3_1_log": a3_1_log,
            "agent3_2_logs": a3_2_logs,
            "agent3_3_logs": a3_3_logs
        }

    new_results = []
    
    def save_intermediate_progress():
        if not new_results:
            return
            
        current_new_pipeline_outputs = [r["pipeline_output"] for r in new_results]
        current_new_a1_logs = [r["agent1_log"] for r in new_results if r["agent1_log"] is not None]
        current_new_a2_logs = [r["agent2_log"] for r in new_results if r["agent2_log"] is not None]
        current_new_a3_1_logs = [r["agent3_1_log"] for r in new_results if r["agent3_1_log"] is not None]
        
        current_new_a3_2_logs = []
        for r in new_results:
            current_new_a3_2_logs.extend(r["agent3_2_logs"])
            
        current_new_a3_3_logs = []
        for r in new_results:
            current_new_a3_3_logs.extend(r["agent3_3_logs"])
            
        # Agent 1
        new_a1_df = pd.DataFrame(current_new_a1_logs)
        dfs_a1 = [df_item for df_item in [existing_a1_df, new_a1_df] if not df_item.empty]
        if dfs_a1:
            pd.concat(dfs_a1, ignore_index=True).to_csv(a1_csv, index=False, encoding="utf-8-sig")
            
        # Agent 2
        if current_new_a2_logs or not existing_a2_df.empty:
            new_a2_df = pd.DataFrame(current_new_a2_logs)
            dfs_a2 = [df_item for df_item in [existing_a2_df, new_a2_df] if not df_item.empty]
            if dfs_a2:
                pd.concat(dfs_a2, ignore_index=True).to_csv(a2_csv, index=False, encoding="utf-8-sig")
                
        # Agent 3.1
        if current_new_a3_1_logs or not existing_a3_1_df.empty:
            new_a3_1_df = pd.DataFrame(current_new_a3_1_logs)
            dfs_a3_1 = [df_item for df_item in [existing_a3_1_df, new_a3_1_df] if not df_item.empty]
            if dfs_a3_1:
                pd.concat(dfs_a3_1, ignore_index=True).to_csv(a3_1_csv, index=False, encoding="utf-8-sig")
                
        # Agent 3.2
        if current_new_a3_2_logs or not existing_a3_2_df.empty:
            new_a3_2_df = pd.DataFrame(current_new_a3_2_logs)
            dfs_a3_2 = [df_item for df_item in [existing_a3_2_df, new_a3_2_df] if not df_item.empty]
            if dfs_a3_2:
                pd.concat(dfs_a3_2, ignore_index=True).to_csv(a3_2_csv, index=False, encoding="utf-8-sig")
                
        # Agent 3.3
        if current_new_a3_3_logs or not existing_a3_3_df.empty:
            new_a3_3_df = pd.DataFrame(current_new_a3_3_logs)
            dfs_a3_3 = [df_item for df_item in [existing_a3_3_df, new_a3_3_df] if not df_item.empty]
            if dfs_a3_3:
                pd.concat(dfs_a3_3, ignore_index=True).to_csv(a3_3_csv, index=False, encoding="utf-8-sig")
                
        # Main pipeline
        new_pipeline_df = pd.DataFrame(current_new_pipeline_outputs)
        new_merged = df[df['synthetic_id'].isin(new_pipeline_df['synthetic_id'])].merge(new_pipeline_df, on="synthetic_id")
        new_merged["cost"] = new_merged.apply(calc_row_cost, axis=1)
        
        dfs_pipeline = [df_item for df_item in [existing_pipeline_df, new_merged] if not df_item.empty]
        if dfs_pipeline:
            pd.concat(dfs_pipeline, ignore_index=True).to_csv(pipeline_csv, index=False, encoding="utf-8-sig")

    if rows_to_process:
        print(f"Processing {len(rows_to_process)} remaining rows out of {len(df)}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_row, row): row for row in rows_to_process}
            count = 0
            for f in as_completed(futures):
                new_results.append(f.result())
                count += 1
                if count % 5 == 0 or count == len(rows_to_process):
                    save_intermediate_progress()
                    print(f"[{config_name}] Saved intermediate progress: {count}/{len(rows_to_process)} rows.")
    else:
        print(f"All rows for {config_name} already evaluated and cached.")
        
    # 4. Load full dataset for E2E metrics evaluation
    merged = pd.read_csv(pipeline_csv)
    
    # Compute Pipeline Metrics
    y_true_cls = merged["gt_is_help_request"].astype(bool)
    y_pred_cls = merged["pred_is_help_request"].astype(bool)
    
    cls_acc = accuracy_score(y_true_cls, y_pred_cls)
    cls_f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    # NER Metrics on true help requests
    df_help = merged[merged["gt_is_help_request"] == True].copy()
    loc_em = 0.0
    v_phone_em = 0.0
    v_name_em = 0.0
    lat_em = 0.0
    
    if not df_help.empty:
        loc_em = np.mean([1 if clean_text_field(r["pred_location_name"]) == clean_text_field(r["gt_location_name"]) else 0 for _, r in df_help.iterrows()])
        v_phone_em = np.mean([1 if clean_phone(r["pred_victim_phone"]) == clean_phone(r["gt_victim_phone"]) else 0 for _, r in df_help.iterrows()])
        v_name_em = np.mean([1 if clean_thai_name(r["pred_victim_name"]) == clean_thai_name(r["gt_victim_name"]) else 0 for _, r in df_help.iterrows()])
        lat_em = np.mean([check_coord_match(r["pred_lat"], r["gt_lat"]) for _, r in df_help.iterrows()])
        
    total_people_gt = 0
    correct_triage = 0
    y_true_all = []
    y_pred_all = []
    
    for _, r in merged.iterrows():
        gt_list = parse_gt_victims(r["gt_victims_json"])
        pred_list = []
        if r["pred_is_help_request"]:
            try:
                pred_list = json.loads(r["pred_victims_list"])
            except:
                pass
                
        total_people_gt += len(gt_list)
        pairs = align_people(gt_list, pred_list)
        for gt_p, pr_p in pairs:
            gt_t = gt_p.get("triage_color") if gt_p else "NONE"
            pr_t = pr_p.get("triage_color") if pr_p else "NONE"
            y_true_all.append(gt_t)
            y_pred_all.append(pr_t)
            if gt_p and pr_p:
                if gt_p.get("triage_color") == pr_p.get("triage_color"):
                    correct_triage += 1
                    
    e2e_triage_acc = correct_triage / total_people_gt if total_people_gt > 0 else 1.0
    e2e_triage_f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0) if y_true_all else 1.0
    
    ci_res = bootstrap_pipeline_metrics(merged)
    
    summary_metrics = {
        "pipeline": config_name,
        "classification_f1": cls_f1,
        "classification_f1_ci": f"{ci_res['classification_f1'][0]:.3f} - {ci_res['classification_f1'][1]:.3f}",
        "classification_accuracy": cls_acc,
        "classification_accuracy_ci": f"{ci_res['classification_accuracy'][0]:.3f} - {ci_res['classification_accuracy'][1]:.3f}",
        "location_em": loc_em,
        "location_em_ci": f"{ci_res['location_em'][0]:.3f} - {ci_res['location_em'][1]:.3f}",
        "victim_phone_em": v_phone_em,
        "victim_phone_em_ci": f"{ci_res['victim_phone_em'][0]:.3f} - {ci_res['victim_phone_em'][1]:.3f}",
        "victim_name_em": v_name_em,
        "lat_em": lat_em,
        "e2e_triage_accuracy": e2e_triage_acc,
        "e2e_triage_accuracy_ci": f"{ci_res['e2e_triage_accuracy'][0]:.3f} - {ci_res['e2e_triage_accuracy'][1]:.3f}",
        "e2e_triage_f1": e2e_triage_f1,
        "e2e_triage_f1_ci": f"{ci_res['e2e_triage_f1'][0]:.3f} - {ci_res['e2e_triage_f1'][1]:.3f}",
        "avg_latency_seconds": merged["latency_seconds"].mean(),
        "total_cost_usd": merged["cost"].sum(),
        "total_input_tokens": int(merged["prompt_tokens"].sum()),
        "total_output_tokens": int(merged["completion_tokens"].sum()),
        "total_tokens": int(merged["prompt_tokens"].sum() + merged["completion_tokens"].sum())
    }
    
    # Save main E2E pipeline result CSV
    merged.to_csv(os.path.join(phase_dir, f"{config_name}_full_pipeline.csv"), index=False, encoding="utf-8-sig")
    
    # Calculate and save sub-agent level F1-scores and metrics for this E2E run
    calculate_and_save_e2e_subagent_metrics(phase_dir, config_name)
    
    return summary_metrics

def generate_comparison_charts(comparison_csv):
    df = pd.read_csv(comparison_csv)
    if df.empty:
        return
        
    print("\nGenerating performance comparison charts...")
    plt.figure(figsize=(12, 8))
    
    # 1. Cost vs Accuracy (Triage)
    plt.subplot(2, 2, 1)
    for _, r in df.iterrows():
        plt.scatter(r["total_cost_usd"], r["e2e_triage_accuracy"], label=r["pipeline"], s=100)
    plt.xlabel("Total Cost (USD)")
    plt.ylabel("E2E Triage Accuracy")
    plt.title("Cost vs Accuracy")
    plt.legend()
    plt.grid(True)
    
    # 2. Latency comparison
    plt.subplot(2, 2, 2)
    plt.bar(df["pipeline"], df["avg_latency_seconds"], color=['blue', 'green', 'orange', 'red'][:len(df)])
    plt.ylabel("Avg Latency (Seconds)")
    plt.title("Pipeline Latency")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    # 3. Location EM Comparison
    plt.subplot(2, 2, 3)
    plt.bar(df["pipeline"], df["location_em"], color=['blue', 'green', 'orange', 'red'][:len(df)])
    plt.ylabel("Location Extract EM")
    plt.title("Location Extraction EM")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    # 4. Token count comparison
    plt.subplot(2, 2, 4)
    plt.bar(df["pipeline"], df["total_tokens"] / 1000.0, color=['blue', 'green', 'orange', 'red'][:len(df)])
    plt.ylabel("Total Tokens (K)")
    plt.title("Total Token Consumption")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "pipeline_performance_charts.png")
    plt.savefig(chart_path)
    print(f"Comparison charts saved to: {chart_path}")
    
    # Save a copy in phase_3 if it has been created
    phase3_dir = os.path.join(RESULTS_DIR, "phase_3")
    if os.path.exists(phase3_dir):
        phase3_chart_path = os.path.join(phase3_dir, "pipeline_performance_charts.png")
        plt.savefig(phase3_chart_path)
        print(f"Comparison charts saved to: {phase3_chart_path}")
        
    plt.close()

def plot_agent1_charts():
    a1_csv = os.path.join(RESULTS_DIR, "phase_1/agent1_classification_comparison.csv")
    if os.path.exists(a1_csv):
        df = pd.read_csv(a1_csv)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="skyblue")
        plt.bar(x + width/2, df["f1"], width, label="F1-score", color="orange")
        plt.xticks(x, df["model"])
        plt.ylabel("Score")
        plt.title("Agent 1 (Classification) Model Comparison")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "phase_1/agent1_classification_charts.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved Agent 1 charts to: {chart_path}")

def plot_agent2_charts():
    a2_csv = os.path.join(RESULTS_DIR, "phase_1/agent2_ner_comparison.csv")
    if os.path.exists(a2_csv):
        df = pd.read_csv(a2_csv)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.25
        plt.bar(x - width, df["location_em"], width, label="Location EM", color="lightgreen")
        plt.bar(x, df["victim_phone_em"], width, label="Phone EM", color="salmon")
        plt.bar(x + width, df["critical_em"], width, label="Critical EM", color="purple")
        plt.xticks(x, df["model"])
        plt.ylabel("Score")
        plt.title("Agent 2 (NER) Model Comparison")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "phase_1/agent2_ner_charts.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved Agent 2 charts to: {chart_path}")

def plot_agent3_1_charts():
    a3_1_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_1_people_extractor_comparison.csv")
    if os.path.exists(a3_1_csv):
        df = pd.read_csv(a3_1_csv)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df["exact_count_match_rate"], width, label="Exact Count Match", color="gold")
        plt.bar(x + width/2, df["age_group_accuracy"], width, label="Age Group Acc", color="teal")
        plt.xticks(x, df["model"])
        plt.ylabel("Score")
        plt.title("Agent 3.1 (People Extractor) Model Comparison")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "phase_1/agent3_1_people_extractor_charts.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved Agent 3.1 charts to: {chart_path}")

def plot_agent3_2_charts():
    a3_2_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_2_pediatric_triage_comparison.csv")
    if os.path.exists(a3_2_csv):
        df = pd.read_csv(a3_2_csv)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="orchid")
        plt.bar(x + width/2, df["f1_score"], width, label="F1-score", color="olive")
        plt.xticks(x, df["model"])
        plt.ylabel("Score")
        plt.title("Agent 3.2 (Pediatric Triage) Model Comparison")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "phase_1/agent3_2_pediatric_triage_charts.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved Agent 3.2 charts to: {chart_path}")

def plot_agent3_3_charts():
    a3_3_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_3_adult_triage_comparison.csv")
    if os.path.exists(a3_3_csv):
        df = pd.read_csv(a3_3_csv)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="pink")
        plt.bar(x + width/2, df["f1_score"], width, label="F1-score", color="brown")
        plt.xticks(x, df["model"])
        plt.ylabel("Score")
        plt.title("Agent 3.3 (Adult Triage) Model Comparison")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "phase_1/agent3_3_adult_triage_charts.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved Agent 3.3 charts to: {chart_path}")

def generate_phase1_charts():
    print("\nGenerating all Phase 1 comparison charts...")
    plot_agent1_charts()
    plot_agent2_charts()
    plot_agent3_1_charts()
    plot_agent3_2_charts()
    plot_agent3_3_charts()

def generate_phase2_charts():
    print("\nGenerating Phase 2 comparison charts...")
    comparison_csv = os.path.join(RESULTS_DIR, "pipeline_comparison_metrics.csv")
    if not os.path.exists(comparison_csv):
        return
        
    df = pd.read_csv(comparison_csv)
    df = df[df["pipeline"].str.startswith("homogeneous_")].copy()
    if df.empty:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width/2, df["classification_f1"], width, label="Classification F1", color="cornflowerblue")
    axes[0].bar(x + width/2, df["e2e_triage_accuracy"], width, label="E2E Triage Acc", color="tomato")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["pipeline"].str.replace("homogeneous_", ""))
    axes[0].set_ylabel("Score")
    axes[0].set_title("E2E Pipelines: Quality Metrics")
    axes[0].legend()
    axes[0].grid(axis="y")
    
    axes[1].bar(x - width/2, df["location_em"], width, label="Location EM", color="mediumseagreen")
    axes[1].bar(x + width/2, df["victim_phone_em"], width, label="Phone EM", color="gold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["pipeline"].str.replace("homogeneous_", ""))
    axes[1].set_ylabel("Score")
    axes[1].set_title("E2E Pipelines: Extraction EM")
    axes[1].legend()
    axes[1].grid(axis="y")
    
    axes[2].bar(df["pipeline"].str.replace("homogeneous_", ""), df["avg_latency_seconds"], color="orchid")
    axes[2].set_ylabel("Avg Latency (Seconds)")
    axes[2].set_title("E2E Pipelines: Latency")
    axes[2].grid(axis="y")
    
    axes[3].bar(df["pipeline"].str.replace("homogeneous_", ""), df["total_cost_usd"], color="salmon")
    axes[3].set_ylabel("Total Cost (USD)")
    axes[3].set_title("E2E Pipelines: Token Cost")
    axes[3].grid(axis="y")
    
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "phase_2/homogeneous_comparison_charts.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Phase 2 comparison charts saved to {chart_path}")

def generate_phase3_comparison_charts(config):
    print("\nGenerating Phase 3 vs Phase 1 comparison charts...")
    p1_scores = {}
    
    # Agent 1
    a1_m = config["agent1"]
    a1_csv = os.path.join(RESULTS_DIR, "phase_1/agent1_classification_comparison.csv")
    if os.path.exists(a1_csv):
        try:
            df = pd.read_csv(a1_csv)
            row = df[df["model"] == a1_m]
            if not row.empty:
                p1_scores["Agent 1 Accuracy"] = row.iloc[0]["accuracy"]
                p1_scores["Agent 1 F1"] = row.iloc[0]["f1"]
        except Exception as e:
            print(f"Error reading Phase 1 Agent 1 data for Phase 3 chart: {e}")
            
    # Agent 2
    a2_m = config["agent2"]
    a2_csv = os.path.join(RESULTS_DIR, "phase_1/agent2_ner_comparison.csv")
    if os.path.exists(a2_csv):
        try:
            df = pd.read_csv(a2_csv)
            row = df[df["model"] == a2_m]
            if not row.empty:
                p1_scores["Agent 2 Location EM"] = row.iloc[0]["location_em"]
                p1_scores["Agent 2 Phone EM"] = row.iloc[0]["victim_phone_em"]
        except Exception as e:
            print(f"Error reading Phase 1 Agent 2 data for Phase 3 chart: {e}")
            
    # Agent 3.1
    a3_1_m = config["agent3_1"]
    a3_1_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_1_people_extractor_comparison.csv")
    if os.path.exists(a3_1_csv):
        try:
            df = pd.read_csv(a3_1_csv)
            row = df[df["model"] == a3_1_m]
            if not row.empty:
                p1_scores["Agent 3.1 Count Match"] = row.iloc[0]["exact_count_match_rate"]
                p1_scores["Agent 3.1 Age Acc"] = row.iloc[0]["age_group_accuracy"]
        except Exception as e:
            print(f"Error reading Phase 1 Agent 3.1 data for Phase 3 chart: {e}")
            
    # Agent 3.2
    a3_2_m = config["agent3_2"]
    a3_2_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_2_pediatric_triage_comparison.csv")
    if os.path.exists(a3_2_csv):
        try:
            df = pd.read_csv(a3_2_csv)
            row = df[df["model"] == a3_2_m]
            if not row.empty:
                p1_scores["Agent 3.2 Pediatric Acc"] = row.iloc[0]["accuracy"]
                p1_scores["Agent 3.2 Pediatric F1"] = row.iloc[0]["f1_score"]
        except Exception as e:
            print(f"Error reading Phase 1 Agent 3.2 data for Phase 3 chart: {e}")
            
    # Agent 3.3
    a3_3_m = config["agent3_3"]
    a3_3_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_3_adult_triage_comparison.csv")
    if os.path.exists(a3_3_csv):
        try:
            df = pd.read_csv(a3_3_csv)
            row = df[df["model"] == a3_3_m]
            if not row.empty:
                p1_scores["Agent 3.3 Adult Acc"] = row.iloc[0]["accuracy"]
                p1_scores["Agent 3.3 Adult F1"] = row.iloc[0]["f1_score"]
        except Exception as e:
            print(f"Error reading Phase 1 Agent 3.3 data for Phase 3 chart: {e}")
            
    p3_scores = {}
    
    # Agent 1
    p3_a1_csv = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_optimized_agent1_classification.csv")
    if os.path.exists(p3_a1_csv):
        try:
            df = pd.read_csv(p3_a1_csv)
            y_true = df["gt_is_help_request"].astype(bool)
            y_pred = df["pred_is_help_request"].astype(bool)
            p3_scores["Agent 1 Accuracy"] = accuracy_score(y_true, y_pred)
            p3_scores["Agent 1 F1"] = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            pass
        
    # Agent 2
    p3_a2_csv = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_optimized_agent2_ner.csv")
    if os.path.exists(p3_a2_csv):
        try:
            df = pd.read_csv(p3_a2_csv)
            p3_scores["Agent 2 Location EM"] = np.mean([1 if clean_text_field(r["pred_location_name"]) == clean_text_field(r["gt_location_name"]) else 0 for _, r in df.iterrows()]) if not df.empty else 0.0
            p3_scores["Agent 2 Phone EM"] = np.mean([1 if clean_phone(r["pred_victim_phone"]) == clean_phone(r["gt_victim_phone"]) else 0 for _, r in df.iterrows()]) if not df.empty else 0.0
        except Exception:
            pass
        
    # Agent 3.1
    p3_a3_1_csv = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_optimized_agent3_1_people_extractor.csv")
    if os.path.exists(p3_a3_1_csv):
        try:
            df = pd.read_csv(p3_a3_1_csv)
            exact_counts = []
            age_group_accs = []
            for _, r in df.iterrows():
                gt_list = parse_gt_victims(r["gt_victims_json"])
                try:
                    pred_list = json.loads(r["pred_people_json"])
                except:
                    pred_list = []
                exact_counts.append(1 if len(gt_list) == len(pred_list) else 0)
                
                pairs = align_people(gt_list, pred_list)
                correct_age_groups = 0
                for gt_p, pr_p in pairs:
                    if gt_p and pr_p:
                        if gt_p.get("age_group") == pr_p.get("age_group"):
                            correct_age_groups += 1
                total_gt = len(gt_list)
                age_group_accs.append(correct_age_groups / total_gt if total_gt > 0 else 1.0)
            p3_scores["Agent 3.1 Count Match"] = np.mean(exact_counts) if exact_counts else 0.0
            p3_scores["Agent 3.1 Age Acc"] = np.mean(age_group_accs) if age_group_accs else 0.0
        except Exception:
            pass
        
    # Agent 3.2
    p3_a3_2_csv = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_optimized_agent3_2_pediatric_triage.csv")
    if os.path.exists(p3_a3_2_csv) and os.path.getsize(p3_a3_2_csv) > 20:
        try:
            df = pd.read_csv(p3_a3_2_csv)
            y_true = df["gt_triage"].fillna("GREEN")
            y_pred = df["pred_triage"].fillna("GREEN")
            p3_scores["Agent 3.2 Pediatric Acc"] = accuracy_score(y_true, y_pred)
            p3_scores["Agent 3.2 Pediatric F1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception:
            p3_scores["Agent 3.2 Pediatric Acc"] = 0.0
            p3_scores["Agent 3.2 Pediatric F1"] = 0.0
            
    # Agent 3.3
    p3_a3_3_csv = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_optimized_agent3_3_adult_triage.csv")
    if os.path.exists(p3_a3_3_csv) and os.path.getsize(p3_a3_3_csv) > 20:
        try:
            df = pd.read_csv(p3_a3_3_csv)
            y_true = df["gt_triage"].fillna("GREEN")
            y_pred = df["pred_triage"].fillna("GREEN")
            p3_scores["Agent 3.3 Adult Acc"] = accuracy_score(y_true, y_pred)
            p3_scores["Agent 3.3 Adult F1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception:
            p3_scores["Agent 3.3 Adult Acc"] = 0.0
            p3_scores["Agent 3.3 Adult F1"] = 0.0
            
    metrics_to_plot = sorted(list(set(p1_scores.keys()).intersection(set(p3_scores.keys()))))
    if not metrics_to_plot:
        print("No matching metrics found between Phase 1 and Phase 3 to plot comparison.")
        return
        
    p1_vals = [p1_scores[m] for m in metrics_to_plot]
    p3_vals = [p3_scores[m] for m in metrics_to_plot]
    
    plt.figure(figsize=(14, 7))
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, p1_vals, width, label="Phase 1 (Isolated Input)", color="royalblue")
    plt.bar(x + width/2, p3_vals, width, label="Phase 3 (Cascaded Pipeline)", color="crimson")
    
    plt.xticks(x, metrics_to_plot, rotation=30, ha="right")
    plt.ylabel("Score / Accuracy")
    plt.title("Error Cascade Comparison: Isolated Phase 1 vs Heterogeneous Pipeline Phase 3")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "phase_3/heterogeneous_vs_phase1_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Phase 3 vs Phase 1 comparison chart saved to {chart_path}")

def calculate_and_save_e2e_subagent_metrics(phase_dir, config_name):
    """
    Reads the detailed sub-agent CSVs for a given E2E config run inside a phase directory,
    calculates their metrics (F1, Accuracy, Latency, Tokens), and appends them to
    a combined sub-agent comparison CSV in that phase directory.
    """
    print(f"Calculating E2E sub-agent F1-scores and metrics for: {config_name}...")
    subagent_metrics = []
    
    # 1. Agent 1 Classification
    a1_csv = os.path.join(phase_dir, f"{config_name}_agent1_classification.csv")
    if os.path.exists(a1_csv):
        try:
            df = pd.read_csv(a1_csv)
            y_true = df["gt_is_help_request"].astype(bool)
            y_pred = df["pred_is_help_request"].astype(bool)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
            f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, zero_division=0)
            
            subagent_metrics.append({
                "pipeline": config_name,
                "agent": "agent1_classification",
                "accuracy": acc,
                "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
                "f1_score": f1,
                "f1_score_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
                "total_input_tokens": int(df["prompt_tokens"].sum()),
                "total_output_tokens": int(df["completion_tokens"].sum()),
                "avg_latency": df["latency"].mean()
            })
        except Exception as e:
            print(f"Error computing E2E Agent 1 metrics: {e}")
            
    # 2. Agent 2 NER
    a2_csv = os.path.join(phase_dir, f"{config_name}_agent2_ner.csv")
    if os.path.exists(a2_csv):
        try:
            df = pd.read_csv(a2_csv)
            loc_matches = [1 if clean_text_field(r["pred_location_name"]) == clean_text_field(r["gt_location_name"]) else 0 for _, r in df.iterrows()]
            phone_matches = [1 if clean_phone(r["pred_victim_phone"]) == clean_phone(r["gt_victim_phone"]) else 0 for _, r in df.iterrows()]
            
            loc_em = np.mean(loc_matches) if loc_matches else 0.0
            phone_em = np.mean(phone_matches) if phone_matches else 0.0
            
            loc_ci = bootstrap_mean_ci(loc_matches)
            phone_ci = bootstrap_mean_ci(phone_matches)
            
            subagent_metrics.append({
                "pipeline": config_name,
                "agent": "agent2_ner",
                "accuracy": loc_em,
                "accuracy_ci": f"{loc_ci[0]:.3f} - {loc_ci[1]:.3f}",
                "f1_score": phone_em,
                "f1_score_ci": f"{phone_ci[0]:.3f} - {phone_ci[1]:.3f}",
                "total_input_tokens": int(df["prompt_tokens"].sum()),
                "total_output_tokens": int(df["completion_tokens"].sum()),
                "avg_latency": df["latency"].mean()
            })
        except Exception as e:
            print(f"Error computing E2E Agent 2 metrics: {e}")
            
    # 3. Agent 3.1 People Extractor
    a3_1_csv = os.path.join(phase_dir, f"{config_name}_agent3_1_people_extractor.csv")
    if os.path.exists(a3_1_csv):
        try:
            df = pd.read_csv(a3_1_csv)
            exact_counts = []
            age_group_accs = []
            for _, r in df.iterrows():
                gt_list = parse_gt_victims(r["gt_victims_json"])
                try:
                    pred_list = json.loads(r["pred_people_json"])
                except:
                    pred_list = []
                exact_counts.append(1 if len(gt_list) == len(pred_list) else 0)
                
                pairs = align_people(gt_list, pred_list)
                correct_age_groups = 0
                for gt_p, pr_p in pairs:
                    if gt_p and pr_p:
                        if gt_p.get("age_group") == pr_p.get("age_group"):
                            correct_age_groups += 1
                total_gt = len(gt_list)
                age_group_accs.append(correct_age_groups / total_gt if total_gt > 0 else 1.0)
                
            count_match = np.mean(exact_counts) if exact_counts else 0.0
            age_acc = np.mean(age_group_accs) if age_group_accs else 0.0
            
            count_ci = bootstrap_mean_ci(exact_counts)
            age_ci = bootstrap_mean_ci(age_group_accs)
            
            subagent_metrics.append({
                "pipeline": config_name,
                "agent": "agent3_1_people_extractor",
                "accuracy": count_match,
                "accuracy_ci": f"{count_ci[0]:.3f} - {count_ci[1]:.3f}",
                "f1_score": age_acc,
                "f1_score_ci": f"{age_ci[0]:.3f} - {age_ci[1]:.3f}",
                "total_input_tokens": int(df["prompt_tokens"].sum()),
                "total_output_tokens": int(df["completion_tokens"].sum()),
                "avg_latency": df["latency"].mean()
            })
        except Exception as e:
            print(f"Error computing E2E Agent 3.1 metrics: {e}")
            
    # 4. Agent 3.2 Pediatric Triage
    a3_2_csv = os.path.join(phase_dir, f"{config_name}_agent3_2_pediatric_triage.csv")
    if os.path.exists(a3_2_csv) and os.path.getsize(a3_2_csv) > 20:
        try:
            df = pd.read_csv(a3_2_csv)
            y_true = df["gt_triage"].fillna("GREEN")
            y_pred = df["pred_triage"].fillna("GREEN")
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
            f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, average='weighted', zero_division=0)
            
            subagent_metrics.append({
                "pipeline": config_name,
                "agent": "agent3_2_pediatric_triage",
                "accuracy": acc,
                "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
                "f1_score": f1,
                "f1_score_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
                "total_input_tokens": int(df["prompt_tokens"].sum()),
                "total_output_tokens": int(df["completion_tokens"].sum()),
                "avg_latency": df["latency"].mean()
            })
        except Exception as e:
            print(f"Error computing E2E Agent 3.2 metrics: {e}")
            
    # 5. Agent 3.3 Adult Triage
    a3_3_csv = os.path.join(phase_dir, f"{config_name}_agent3_3_adult_triage.csv")
    if os.path.exists(a3_3_csv) and os.path.getsize(a3_3_csv) > 20:
        try:
            df = pd.read_csv(a3_3_csv)
            y_true = df["gt_triage"].fillna("GREEN")
            y_pred = df["pred_triage"].fillna("GREEN")
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            acc_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), accuracy_score)
            f1_ci = bootstrap_ci(y_true.tolist(), y_pred.tolist(), f1_score, average='weighted', zero_division=0)
            
            subagent_metrics.append({
                "pipeline": config_name,
                "agent": "agent3_3_adult_triage",
                "accuracy": acc,
                "accuracy_ci": f"{acc_ci[0]:.3f} - {acc_ci[1]:.3f}",
                "f1_score": f1,
                "f1_score_ci": f"{f1_ci[0]:.3f} - {f1_ci[1]:.3f}",
                "total_input_tokens": int(df["prompt_tokens"].sum()),
                "total_output_tokens": int(df["completion_tokens"].sum()),
                "avg_latency": df["latency"].mean()
            })
        except Exception as e:
            print(f"Error computing E2E Agent 3.3 metrics: {e}")
            
    if subagent_metrics:
        out_df = pd.DataFrame(subagent_metrics)
        comp_csv = os.path.join(phase_dir, "subagent_comparison_metrics.csv")
        if os.path.exists(comp_csv):
            try:
                old_df = pd.read_csv(comp_csv)
                old_df = old_df[old_df["pipeline"] != config_name]
                out_df = pd.concat([old_df, out_df], ignore_index=True)
            except Exception:
                pass
        out_df.to_csv(comp_csv, index=False)
        print(f"Saved E2E sub-agent metrics to: {comp_csv}")

def get_best_hetero_config():
    """
    Reads Phase 1 comparison files to find the best model for each agent.
    If files don't exist, falls back to HETERO_DEFAULT_CONFIG.
    """
    best_config = HETERO_DEFAULT_CONFIG.copy()
    print("\nCalculating optimal Heterogeneous config from Phase 1 F1-scores...")
    
    # 1. Agent 1 (Classification)
    a1_csv = os.path.join(RESULTS_DIR, "phase_1/agent1_classification_comparison.csv")
    if os.path.exists(a1_csv):
        try:
            df = pd.read_csv(a1_csv)
            best_model = df.sort_values(by="f1", ascending=False).iloc[0]["model"]
            best_config["agent1"] = best_model
            print(f"-> Agent 1: Selected {best_model} (Highest F1)")
        except Exception as e:
            print(f"Error reading Agent 1 comparison: {e}")
            
    # 2. Agent 2 (NER)
    a2_csv = os.path.join(RESULTS_DIR, "phase_1/agent2_ner_comparison.csv")
    if os.path.exists(a2_csv):
        try:
            df = pd.read_csv(a2_csv)
            # Use location_em + phone_em + critical_em average as proxy for extraction F1
            df["combined"] = (df["location_em"] + df["victim_phone_em"] + df["critical_em"]) / 3.0
            best_model = df.sort_values(by="combined", ascending=False).iloc[0]["model"]
            best_config["agent2"] = best_model
            print(f"-> Agent 2: Selected {best_model} (Highest combined EM)")
        except Exception as e:
            print(f"Error reading Agent 2 comparison: {e}")
            
    # 3. Agent 3.1 (People Extractor)
    a3_1_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_1_people_extractor_comparison.csv")
    if os.path.exists(a3_1_csv):
        try:
            df = pd.read_csv(a3_1_csv)
            best_model = df.sort_values(by="age_group_accuracy", ascending=False).iloc[0]["model"]
            best_config["agent3_1"] = best_model
            print(f"-> Agent 3.1: Selected {best_model} (Highest age group accuracy)")
        except Exception as e:
            print(f"Error reading Agent 3.1 comparison: {e}")
            
    # 4. Agent 3.2 (Pediatric Triage)
    a3_2_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_2_pediatric_triage_comparison.csv")
    if os.path.exists(a3_2_csv):
        try:
            df = pd.read_csv(a3_2_csv)
            best_model = df.sort_values(by="f1_score", ascending=False).iloc[0]["model"]
            best_config["agent3_2"] = best_model
            print(f"-> Agent 3.2: Selected {best_model} (Highest F1-score)")
        except Exception as e:
            print(f"Error reading Agent 3.2 comparison: {e}")
            
    # 5. Agent 3.3 (Adult Triage)
    a3_3_csv = os.path.join(RESULTS_DIR, "phase_1/agent3_3_adult_triage_comparison.csv")
    if os.path.exists(a3_3_csv):
        try:
            df = pd.read_csv(a3_3_csv)
            best_model = df.sort_values(by="f1_score", ascending=False).iloc[0]["model"]
            best_config["agent3_3"] = best_model
            print(f"-> Agent 3.3: Selected {best_model} (Highest F1-score)")
        except Exception as e:
            print(f"Error reading Agent 3.3 comparison: {e}")
            
    return best_config

def main():
    parser = argparse.ArgumentParser(description="Evaluate Disaster Multi-Agent Processing Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate")
    parser.add_argument("--phases", type=str, default="1,2,3", help="Phases to run (1: Individual, 2: Homogeneous, 3: Heterogeneous)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers for ThreadPoolExecutor")
    args = parser.parse_args()
    
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}!")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_PATH)
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} rows.")
        df = df.head(args.limit)
        
    print(f"Loaded {len(df)} rows.")
    
    phases = [int(p.strip()) for p in args.phases.split(",")]
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    
    # Phase 1: Individual Agent Evaluation
    if 1 in phases:
        run_phase1_classification(df, models, args.workers)
        plot_agent1_charts()
        
        run_phase1_ner(df, models, args.workers)
        plot_agent2_charts()
        
        run_phase1_people_extractor(df, models, args.workers)
        plot_agent3_1_charts()
        
        run_phase1_triage(df, models, args.workers)
        plot_agent3_2_charts()
        plot_agent3_3_charts()
        
    pipeline_results = []
    
    # Phase 2: Homogeneous Pipeline Evaluation
    if 2 in phases:
        for model in models:
            config = {
                "agent1": model,
                "agent2": model,
                "agent3_1": model,
                "agent3_2": model,
                "agent3_3": model
            }
            res = run_pipeline_e2e(df, config, f"homogeneous_{model}", args.workers, phase_num=2)
            pipeline_results.append(res)
            
            # Save/update comparison results immediately
            comparison_df = pd.DataFrame(pipeline_results)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            comparison_csv = os.path.join(RESULTS_DIR, "pipeline_comparison_metrics.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            print(f"Updated pipeline comparison metrics to: {comparison_csv}")
            
            # Generate phase 2 charts immediately
            generate_phase2_charts()
            
    # Phase 3: Heterogeneous Pipeline Evaluation
    if 3 in phases:
        best_config = get_best_hetero_config()
        print(f"Using Dynamic Heterogeneous Config: {best_config}")
        res = run_pipeline_e2e(df, best_config, "heterogeneous_optimized", args.workers, phase_num=3)
        pipeline_results.append(res)
        
        # Save/update comparison results immediately
        comparison_df = pd.DataFrame(pipeline_results)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        comparison_csv = os.path.join(RESULTS_DIR, "pipeline_comparison_metrics.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Updated pipeline comparison metrics to: {comparison_csv}")
        
        generate_phase3_comparison_charts(best_config)
        
    if pipeline_results:
        print("\nFinal Pipeline Comparison summary:")
        comparison_df = pd.DataFrame(pipeline_results)
        print(comparison_df[["pipeline", "classification_f1", "classification_f1_ci", "e2e_triage_f1", "e2e_triage_f1_ci", "e2e_triage_accuracy", "e2e_triage_accuracy_ci", "avg_latency_seconds", "total_cost_usd"]].to_markdown(index=False))
        
        # Generate final E2E charts
        generate_comparison_charts(os.path.join(RESULTS_DIR, "pipeline_comparison_metrics.csv"))

if __name__ == "__main__":
    main()
