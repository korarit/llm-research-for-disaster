import os
import pandas as pd

def check_model(model_name, outfile):
    csv_path = f"e:/nlp-for-disaster/exp4/results/{model_name}_results.csv"
    if not os.path.exists(csv_path):
        outfile.write(f"File not found: {csv_path}\n")
        return
        
    df = pd.read_csv(csv_path)
    df_true = df[df['gt_is_help_request'] == True].copy()
    
    mismatch_victim = df_true[df_true['pred_victim_name'].astype(str).str.lower().str.strip() != df_true['gt_victim_name'].astype(str).str.lower().str.strip()]
    
    outfile.write(f"=== {model_name} (Total Victim Name mismatches: {len(mismatch_victim)}/{len(df_true)}) ===\n")
    
    for idx, r in mismatch_victim.head(10).iterrows():
        outfile.write(f"ID: {r['synthetic_id']}\n")
        outfile.write(f"Text: {r['generated_text']}\n")
        outfile.write(f"GT Victim: {r['gt_victim_name']}\n")
        outfile.write(f"Pred Victim: {r['pred_victim_name']}\n")
        outfile.write(f"GT Reporter: {r['gt_reporter_name']}\n")
        outfile.write(f"Pred Reporter: {r['pred_reporter_name']}\n")
        outfile.write("-" * 50 + "\n")

def main():
    out_path = "e:/nlp-for-disaster/scratch/check_mismatch.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        check_model("gemma-4", f)
        f.write("\n" + "="*80 + "\n\n")
        check_model("deepseek-v4-flash", f)

if __name__ == '__main__':
    main()
