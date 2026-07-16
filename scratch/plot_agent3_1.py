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
    # Read the CSV file
    csv_path = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\agent3_1_people_extractor_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['model'])
    
    # Destination directories
    primary_dir = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\ner_graph"
    os.makedirs(primary_dir, exist_ok=True)
    
    secondary_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if secondary_dir:
        os.makedirs(secondary_dir, exist_ok=True)
        
    # Translation dictionary for metrics
    metric_translation = {
        "exact_count_match_rate": "ความถูกต้องในการระบุจำนวนคน (ตรงทั้งหมด)",
        "age_group_accuracy": "ความถูกต้องของการระบุช่วงกลุ่มอายุ"
    }
    
    cols = ["exact_count_match_rate", "age_group_accuracy"]
    
    plt.figure(figsize=(9, 6))
    
    # Melt the dataframe for seaborn plotting
    cols_to_keep = ['model'] + cols
    melted_df = df[cols_to_keep].melt(id_vars='model', var_name='metric', value_name='score')
    
    # Translate metric names
    melted_df['metric_thai'] = melted_df['metric'].map(metric_translation).fillna(melted_df['metric'])
    
    ax = sns.barplot(
        data=melted_df,
        x='metric_thai',
        y='score',
        hue='model',
        palette='viridis'
    )
    
    plt.title("ความถูกต้องของการระบุจำนวนและช่วงอายุผู้ประสบภัย (Agent 3.1 People Extractor)", pad=40, fontweight='bold')
    plt.ylabel("คะแนนความถูกต้อง")
    plt.xlabel("ประเภทของข้อมูลที่ดึงออกมา")
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
    
    filename = "agent3_1_extractor_comparison.png"
    
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
