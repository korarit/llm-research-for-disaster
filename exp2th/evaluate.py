import os
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_chart(metrics_path, output_path, title):
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return
    df = pd.read_csv(metrics_path)
    if df.empty:
        return
        
    df_melted = df.melt(id_vars=['model', 'temperature'],
                         value_vars=['text_informativeness_f1', 'text_category_f1'],
                         var_name='metric', value_name='f1_score')
                         
    df_melted['metric'] = df_melted['metric'].replace({
        'text_informativeness_f1': 'Informativeness F1',
        'text_category_f1': 'Category F1'
    })
    
    temp_vals = sorted(df_melted['temperature'].unique())
    temp_labels = [f'T={t}' for t in temp_vals]
    
    g = sns.catplot(
        data=df_melted, kind='bar',
        x='model', y='f1_score', hue='temperature',
        col='metric', palette='Set2',
        height=6, aspect=0.8,
        hue_order=temp_vals,
        legend_out=True
    )
    g.set_axis_labels('Model', 'F1 Score')
    g.set_titles('{col_name}')
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    for t, l in zip(g._legend.texts, temp_labels):
        t.set_text(l)
        
    for ax in g.axes.flat:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=8)
        ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    results_dir = "e:/nlp-for-disaster/exp2th/results"
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    comparison_metrics = []
    
    os.makedirs(os.path.join(results_dir, "confusion_matrices"), exist_ok=True)
    
    for model_name in models:
        csv_path = os.path.join(results_dir, f"{model_name}_temp_results_th.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_dir, f"{model_name}_results_th.csv")
            if not os.path.exists(csv_path):
                print(f"Result file for {model_name} not found.")
                continue
            df = pd.read_csv(csv_path)
            if 'temperature' not in df.columns:
                df['temperature'] = 0.0
        else:
            df = pd.read_csv(csv_path)
            
        temperatures = sorted(df['temperature'].unique())
        
        for temp in temperatures:
            sub_df = df[df['temperature'] == temp]
            
            # Informativeness F1
            y_true_info = sub_df['true_text_info']
            y_pred_info = sub_df['predicted_info']
            
            info_f1 = f1_score(
                y_true_info, y_pred_info,
                pos_label="informative",
                average="binary"
            )
            
            # Category F1
            y_true_human = sub_df['true_text_human']
            y_pred_human = sub_df['predicted_category']
            
            category_f1 = f1_score(
                y_true_human, y_pred_human,
                average="weighted"
            )
            
            comparison_metrics.append({
                "model": model_name,
                "temperature": temp,
                "text_informativeness_f1": info_f1,
                "text_category_f1": category_f1,
                "avg_latency_seconds": sub_df['latency_seconds'].mean() if 'latency_seconds' in sub_df.columns else 0.0,
                "total_tokens_used": (sub_df['token_in_use'].sum() + sub_df['token_out_use'].sum()) if 'token_in_use' in sub_df.columns else 0
            })
            
            # Draw Confusion Matrix for Category
            labels = sorted(list(set(y_true_human.unique()).union(set(y_pred_human.unique()))))
            cm = confusion_matrix(y_true_human, y_pred_human, labels=labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix (Category) - {model_name} Temp {temp} (Exp 2TH)")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(results_dir, f"confusion_matrices/{model_name.split('-')[0]}_temp_{temp}_confusion_matrix_th.png")
            plt.savefig(cm_path)
            plt.close()
            
    metrics_df = pd.DataFrame(comparison_metrics)
    metrics_csv = os.path.join(results_dir, "th_model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved Exp 2TH comparison metrics to {metrics_csv}")
    print(metrics_df)
    
    # Generate chart
    chart_path = os.path.join(results_dir, "th_model_comparison_chart.png")
    generate_chart(metrics_csv, chart_path, "Thai Two-Layer Joint (Exp 2TH) F1 Comparison")
    print(f"Generated chart at {chart_path}")
    
    # Generate Exp 1 vs Exp 2 comparison
    exp1_metrics_csv = "e:/nlp-for-disaster/exp1th/results/th_model_comparison_metrics.csv"
    if os.path.exists(exp1_metrics_csv) and not metrics_df.empty:
        print("Found Exp 1TH metrics, generating Exp 1TH vs Exp 2TH comparison...")
        exp1_df = pd.read_csv(exp1_metrics_csv)
        
        # Merge metrics on model and temperature
        merged = pd.merge(
            exp1_df, metrics_df,
            on=['model', 'temperature'],
            suffixes=('_exp1th', '_exp2th')
        )
        
        comparison_csv = os.path.join(results_dir, "th_exp1_vs_exp2_comparison.csv")
        merged.to_csv(comparison_csv, index=False)
        print(f"Saved comparison to {comparison_csv}")
    else:
        print("Exp 1TH metrics not found. Skipping cross-architecture comparison.")

if __name__ == '__main__':
    main()
