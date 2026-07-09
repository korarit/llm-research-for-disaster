import os
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_dir = "e:/nlp-for-disaster/exp1E/results"
    models = ["deepseek-v4-flash", "typhoon-v2.5", "gemma-4"]
    comparison_metrics = []
    
    os.makedirs(os.path.join(results_dir, "confusion_matrices"), exist_ok=True)
    
    for model_name in models:
        csv_path = os.path.join(results_dir, f"{model_name}_temp_results.csv")
        if not os.path.exists(csv_path):
            print(f"Result file {csv_path} not found.")
            continue
            
        df = pd.read_csv(csv_path)
        temperatures = sorted(df['temperature'].unique())
        
        for temp in temperatures:
            sub_df = df[df['temperature'] == temp]
            
            # Informativeness F1
            y_true_info = sub_df['true_text_info']
            y_pred_info = sub_df['mapped_predicted_info']
            
            info_f1 = f1_score(
                y_true_info, y_pred_info,
                pos_label="informative",
                average="binary"
            )
            
            # Category F1
            y_true_human = sub_df['true_text_human']
            y_pred_human = sub_df['mapped_predicted_category']
            
            category_f1 = f1_score(
                y_true_human, y_pred_human,
                average="weighted"
            )
            
            comparison_metrics.append({
                "model": model_name,
                "temperature": temp,
                "text_informativeness_f1": info_f1,
                "text_category_f1": category_f1,
                "avg_latency_seconds": sub_df['latency_seconds'].mean(),
                "total_tokens_used": sub_df['token_in_use'].sum() + sub_df['token_out_use'].sum()
            })
            
            # Draw Confusion Matrix for Category
            labels = sorted(list(set(y_true_human.unique()).union(set(y_pred_human.unique()))))
            cm = confusion_matrix(y_true_human, y_pred_human, labels=labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix (Category) - {model_name} Temp {temp} (Exp 1E)")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(results_dir, f"confusion_matrices/{model_name.split('-')[0]}_temp_{temp}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            
    metrics_df = pd.DataFrame(comparison_metrics)
    metrics_csv = os.path.join(results_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved Exp 1E comparison metrics to {metrics_csv}")
    print(metrics_df)

if __name__ == '__main__':
    main()
