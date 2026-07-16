import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and use a Thai-supporting font
# Leelawadee UI is standard on Windows and supports Thai characters beautifully.
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'Leelawadee UI',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def main():
    # Read the CSV file
    csv_path = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\agent2_ner_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['model'])
    
    # Destination directories
    primary_dir = r"e:\nlp-for-disaster\exp_full_agent\results\phase_1\ner_graph"
    os.makedirs(primary_dir, exist_ok=True)
    
    # Check if a secondary dir (like artifact dir) is passed
    secondary_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if secondary_dir:
        os.makedirs(secondary_dir, exist_ok=True)
        
    # Translation dictionary for metrics
    metric_translation = {
        "location": "ชื่อสถานที่",
        "map_url": "ลิงก์แผนที่",
        "lat": "ละติจูด",
        "lng": "ลองจิจูด",
        "victim_name": "ชื่อผู้ประสบภัย",
        "victim_phone": "เบอร์โทรผู้ประสบภัย",
        "reporter_name": "ชื่อผู้แจ้งเหตุ",
        "reporter_phone": "เบอร์โทรผู้แจ้งเหตุ",
        "dead": "เสียชีวิต",
        "critical": "ประสบภัยขั้นวิกฤต",
        "urgent": "เร่งด่วน",
        "safe": "ปลอดภัย",
        "child": "เด็ก",
        "bedridden": "ผู้ป่วยติดเตียง",
        "firstaid": "ปฐมพยาบาล",
        "food": "อาหารและน้ำ",
        "energy": "พลังงาน/ไฟฟ้า"
    }
        
    # Define groups
    groups = {
        "location_info": {
            "title": "ความถูกต้องของการดึงข้อมูลสถานที่และพิกัด (Exact Match)",
            "cols": ["location_em", "map_url_em", "lat_em", "lng_em"],
            "filename": "location_info_comparison.png"
        },
        "contact_info": {
            "title": "ความถูกต้องของการดึงข้อมูลติดต่อและชื่อบุคคล (Exact Match)",
            "cols": ["victim_name_em", "victim_phone_em", "reporter_name_em", "reporter_phone_em"],
            "filename": "contact_info_comparison.png"
        },
        "victim_status": {
            "title": "ความถูกต้องของการสกัดสถานะความรุนแรงและกลุ่มเปราะบาง (Exact Match)",
            "cols": ["dead_em", "critical_em", "urgent_em", "safe_em", "child_em", "bedridden_em"],
            "filename": "victim_status_comparison.png"
        },
        "needs_info": {
            "title": "ความถูกต้องของการวิเคราะห์ประเภทความช่วยเหลือที่ต้องการ (Exact Match)",
            "cols": ["firstaid_em", "food_em", "energy_em"],
            "filename": "needs_comparison.png"
        }
    }
    
    # Plot 1-4: The EM Groups
    for gkey, ginfo in groups.items():
        plt.figure(figsize=(10, 6.5))
        
        cols_to_keep = ['model'] + ginfo['cols']
        melted_df = df[cols_to_keep].melt(id_vars='model', var_name='metric', value_name='score')
        
        # Clean metric name and translate
        melted_df['metric_clean'] = melted_df['metric'].str.replace('_em', '')
        melted_df['metric_thai'] = melted_df['metric_clean'].map(metric_translation).fillna(melted_df['metric_clean'])
        
        ax = sns.barplot(
            data=melted_df,
            x='metric_thai',
            y='score',
            hue='model',
            palette='viridis'
        )
        
        # Title with extra padding to fit the top legend
        plt.title(ginfo['title'], pad=40, fontweight='bold')
        plt.ylabel("คะแนนความถูกต้อง (Exact Match)")
        plt.xlabel("ประเภทข้อมูล")
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
                            fontsize=9,
                            fontweight='bold')
                            
        plt.tight_layout()
        
        # Save to primary
        save_path_primary = os.path.join(primary_dir, ginfo['filename'])
        plt.savefig(save_path_primary, dpi=150)
        print(f"Saved {ginfo['filename']} to {primary_dir}")
        
        # Save to secondary if exists
        if secondary_dir:
            save_path_secondary = os.path.join(secondary_dir, ginfo['filename'])
            plt.savefig(save_path_secondary, dpi=150)
            print(f"Saved {ginfo['filename']} to secondary {secondary_dir}")
            
        plt.close()
        
    # Plot 5: Average Latency
    plt.figure(figsize=(8, 5.5))
    ax = sns.barplot(
        data=df,
        x='model',
        y='avg_latency',
        palette='magma'
    )
    plt.title("เวลาตอบสนองเฉลี่ย (วินาที) - ค่าน้อยกว่าคือดีกว่า", pad=15, fontweight='bold')
    plt.ylabel("เวลาตอบสนอง (วินาที)")
    plt.xlabel("โมเดล")
    plt.ylim(0, df['avg_latency'].max() * 1.15)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f} วินาที',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 6),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold')
                        
    plt.tight_layout()
    lat_filename = "avg_latency_comparison.png"
    
    # Save to primary
    save_path_primary = os.path.join(primary_dir, lat_filename)
    plt.savefig(save_path_primary, dpi=150)
    print(f"Saved {lat_filename} to {primary_dir}")
    
    # Save to secondary if exists
    if secondary_dir:
        save_path_secondary = os.path.join(secondary_dir, lat_filename)
        plt.savefig(save_path_secondary, dpi=150)
        print(f"Saved {lat_filename} to secondary {secondary_dir}")
        
    plt.close()

if __name__ == '__main__':
    main()
