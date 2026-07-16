import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Set style and use a Thai-supporting font
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'Leelawadee UI',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def bootstrap_class_f1_ci(y_true, y_pred, label, n_iterations=1000, alpha=0.95):
    n_samples = len(y_true)
    y_true_bin = (y_true == label).astype(int)
    y_pred_bin = (y_pred == label).astype(int)
    
    # Calculate point estimate
    point_est = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    boot_scores = []
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = f1_score(y_true_bin[indices], y_pred_bin[indices], zero_division=0)
        boot_scores.append(score)
        
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    ci_lower = np.percentile(boot_scores, lower_p)
    ci_upper = np.percentile(boot_scores, upper_p)
    
    return point_est, ci_lower, ci_upper

def process_and_plot(triage_type, file_pattern, title, filename, out_dir, secondary_dir=None):
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    classes = ["GREEN", "YELLOW", "RED"]
    class_thai = {
        "GREEN": "ปลอดภัย (เขียว)",
        "YELLOW": "มีการบาดเจ็บ (เหลือง)",
        "RED": "บาดเจ็บสาหัส (แดง)"
    }
    class_colors = {
        "GREEN": "#27ae60",   # Emerald green
        "YELLOW": "#f39c12",  # Amber/dark yellow (for visibility)
        "RED": "#e74c3c"      # Alizarin red
    }
    
    # Store results
    # {class: {model: (point, ci_lower, ci_upper)}}
    results = {c: {} for c in classes}
    
    results_base_dir = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1"
    
    for model in models:
        csv_name = f"agent3_{triage_type}_triage_{model}.csv"
        csv_path = os.path.join(results_base_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found.")
            sys.exit(1)
            
        df = pd.read_csv(csv_path)
        # Clean labels
        y_true = df['gt_triage'].str.upper().str.strip().values
        y_pred = df['pred_triage'].fillna("UNKNOWN").str.upper().str.strip().values
        
        for c in classes:
            point, lower, upper = bootstrap_class_f1_ci(y_true, y_pred, c)
            results[c][model] = (point, lower, upper)
            print(f"[{triage_type.upper()}] Model: {model}, Class: {c} -> F1: {point:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
            
    # Now set up grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    x = np.arange(len(classes))
    width = 0.25
    positions = {
        "deepseek-v4-flash": x - width,
        "typhoon-v2.5": x,
        "gemma-4": x + width
    }
    
    model_colors = sns.color_palette("viridis", len(models))
    
    for idx, model in enumerate(models):
        points = [results[c][model][0] for c in classes]
        lowers = [results[c][model][1] for c in classes]
        uppers = [results[c][model][2] for c in classes]
        
        rects = ax.bar(
            positions[model],
            points,
            width,
            label=model,
            color=model_colors[idx]
        )
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 8),  # 8 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9.5,
                fontweight='bold'
            )
            
    # Style chart
    ax.set_title(title, pad=40, fontweight='bold')
    ax.set_ylabel("คะแนน F1-Score (พร้อม 95% Confidence Interval)")
    ax.set_xlabel("ระดับความรุนแรงของอาการ")
    ax.set_xticks(x)
    ax.set_xticklabels([class_thai[c] for c in classes])
    ax.set_ylim(0, 1.15)
    
    # Legend at the top horizontal
    ax.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.01), 
        ncol=3, 
        frameon=True,
        facecolor='white',
        edgecolor='none'
    )
    
    plt.tight_layout()
    
    # Save files
    primary_path = os.path.join(out_dir, filename)
    plt.savefig(primary_path, dpi=150)
    print(f"Saved {filename} to {out_dir}")
    
    if secondary_dir:
        secondary_path = os.path.join(secondary_dir, filename)
        plt.savefig(secondary_path, dpi=150)
        print(f"Saved {filename} to secondary {secondary_dir}")
        
    plt.close()

def main():
    primary_dir = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\ner_graph"
    os.makedirs(primary_dir, exist_ok=True)
    
    secondary_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if secondary_dir:
        os.makedirs(secondary_dir, exist_ok=True)
        
    # Process Pediatric Triage (2)
    process_and_plot(
        triage_type="2_pediatric",
        file_pattern="agent3_2_pediatric_triage_*.csv",
        title="ความสามารถในการแยกแยะความรุนแรงผู้ประสบภัยเด็ก (อายุ < 12 ปี)",
        filename="agent3_pediatric_class_f1_comparison.png",
        out_dir=primary_dir,
        secondary_dir=secondary_dir
    )
    
    # Process Adult Triage (3)
    process_and_plot(
        triage_type="3_adult",
        file_pattern="agent3_3_adult_triage_*.csv",
        title="ความสามารถในการแยกแยะความรุนแรงผู้ประสบภัยผู้ใหญ่ (อายุ >= 12 ปี)",
        filename="agent3_adult_class_f1_comparison.png",
        out_dir=primary_dir,
        secondary_dir=secondary_dir
    )

if __name__ == '__main__':
    main()
