import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def main():
    # File paths
    pediatric_path = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\agent3_2_pediatric_triage_comparison.csv"
    adult_path = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\agent3_3_adult_triage_comparison.csv"
    
    if not os.path.exists(pediatric_path) or not os.path.exists(adult_path):
        print("Error: One or both CSV files not found.")
        sys.exit(1)
        
    # Read CSVs
    df_p = pd.read_csv(pediatric_path).dropna(subset=['model'])
    df_a = pd.read_csv(adult_path).dropna(subset=['model'])
    
    # Extract model and f1_score
    df_p = df_p[['model', 'f1_score']].copy()
    df_p['group'] = "อายุ < 12 ปี (ผู้ป่วยเด็ก)"
    
    df_a = df_a[['model', 'f1_score']].copy()
    df_a['group'] = "อายุ >= 12 ปี (ผู้ป่วยผู้ใหญ่)"
    
    # Combine data
    df_combined = pd.concat([df_a, df_p], ignore_index=True)
    
    print("Combined Data:")
    print(df_combined)
    
    # Destination directories
    primary_dir = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\ner_graph"
    os.makedirs(primary_dir, exist_ok=True)
    
    secondary_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if secondary_dir:
        os.makedirs(secondary_dir, exist_ok=True)
        
    # Plotting
    plt.figure(figsize=(9, 6))
    
    ax = sns.barplot(
        data=df_combined,
        x='group',
        y='f1_score',
        hue='model',
        palette='viridis'
    )
    
    plt.title("การเปรียบเทียบค่า F1-Score ของการคัดแยกประเภทผู้ป่วย (Triage Classification)", pad=40, fontweight='bold')
    plt.ylabel("คะแนน F1-Score")
    plt.xlabel("กลุ่มอายุผู้ป่วย")
    plt.ylim(0, 1.1)
    
    # Place legend horizontally at the top
    plt.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.01), 
        ncol=3, 
        frameon=True,
        facecolor='white',
        edgecolor='none'
    )
    
    # Add labels on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 6),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold')
                        
    plt.tight_layout()
    
    filename = "agent3_triage_f1_comparison.png"
    
    # Save to primary
    save_path_primary = os.path.join(primary_dir, filename)
    plt.savefig(save_path_primary, dpi=150)
    print(f"Saved {filename} to {primary_dir}")
    
    # Save to secondary if exists
    if secondary_dir:
        save_path_secondary = os.path.join(secondary_dir, filename)
        plt.savefig(save_path_secondary, dpi=150)
        print(f"Saved {filename} to secondary {secondary_dir}")
        
    plt.close()

if __name__ == '__main__':
    main()
