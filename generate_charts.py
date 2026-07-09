import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_exp_chart(metrics_path, output_path, title):
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return
    df = pd.read_csv(metrics_path)

    has_temp = 'temperature' in df.columns

    id_vars = ['model']
    if has_temp:
        id_vars.append('temperature')

    df_melted = df.melt(id_vars=id_vars,
                         value_vars=['text_informativeness_f1', 'text_category_f1'],
                         var_name='metric', value_name='f1_score')

    df_melted['metric'] = df_melted['metric'].replace({
        'text_informativeness_f1': 'Informativeness F1',
        'text_category_f1': 'Category F1'
    })

    if has_temp:
        df_melted['temperature'] = df_melted['temperature'].astype(float)
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

        for new_labels in [[t.get_text() for t in g._legend.texts], temp_labels]:
            pass
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
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_melted, x='model', y='f1_score', hue='metric', ax=ax1, palette='Set2')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylim(0, 1.0)
        for p in ax1.patches:
            height = p.get_height()
            if height > 0:
                ax1.annotate(f'{height:.3f}',
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points',
                             fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    print(f"Saved experiment comparison chart to {output_path}")

def generate_cross_arch_chart(cross_csv_path, output_path):
    if not os.path.exists(cross_csv_path):
        print(f"Cross-architecture metrics file {cross_csv_path} not found.")
        return
    df = pd.read_csv(cross_csv_path)
    
    # Plot Category F1 across architectures
    plt.figure(figsize=(12, 7))
    
    df_melted = df.melt(id_vars=['model'], value_vars=['exp1_cat_f1', 'exp2_cat_f1', 'exp3_cat_f1'],
                         var_name='architecture', value_name='category_f1')
    
    df_melted['architecture'] = df_melted['architecture'].replace({
        'exp1_cat_f1': 'Exp 1: Single-Layer (Flat)',
        'exp2_cat_f1': 'Exp 2: Two-Layer Joint',
        'exp3_cat_f1': 'Exp 3: 2-Agent Sequential'
    })
    
    ax = sns.barplot(data=df_melted, x='model', y='category_f1', hue='architecture', palette='viridis')
    plt.title('Category F1-Score Comparison Across Architectures', fontsize=14, fontweight='bold')
    plt.ylabel('Weighted F1-Score (Category)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1.0)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center',
                         xytext=(0, 9),
                         textcoords='offset points',
                         fontsize=9)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved cross-architecture comparison chart to {output_path}")

def generate_latency_comparison(cross_csv_path, output_path):
    if not os.path.exists(cross_csv_path):
        return
    df = pd.read_csv(cross_csv_path)
    
    # Plot Latency across architectures
    plt.figure(figsize=(12, 7))
    
    df_melted = df.melt(id_vars=['model'], value_vars=['exp1_latency_seconds', 'exp2_latency_seconds', 'exp3_latency_seconds'],
                         var_name='architecture', value_name='avg_latency')
    
    df_melted['architecture'] = df_melted['architecture'].replace({
        'exp1_latency_seconds': 'Exp 1: Single-Layer (Flat)',
        'exp2_latency_seconds': 'Exp 2: Two-Layer Joint',
        'exp3_latency_seconds': 'Exp 3: 2-Agent Sequential'
    })
    
    ax = sns.barplot(data=df_melted, x='model', y='avg_latency', hue='architecture', palette='magma')
    plt.title('Average Latency per Row Comparison Across Architectures', fontsize=14, fontweight='bold')
    plt.ylabel('Average Latency (seconds)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}s',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center',
                         xytext=(0, 9),
                         textcoords='offset points',
                         fontsize=9)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved latency comparison chart to {output_path}")

def main():
    print("Generating charts...")
    # Exp 1 chart
    generate_exp_chart(
        "e:/nlp-for-disaster/exp1/results/model_comparison_metrics.csv",
        "e:/nlp-for-disaster/exp1/results/model_comparison_chart.png",
        "Experiment 1: Flat Classification F1-Score Comparison"
    )
    
    # Exp 2 chart
    generate_exp_chart(
        "e:/nlp-for-disaster/exp2/results/model_comparison_metrics.csv",
        "e:/nlp-for-disaster/exp2/results/model_comparison_chart.png",
        "Experiment 2: Two-Layer Joint Classification F1-Score Comparison"
    )
    
    # Exp 3 chart
    generate_exp_chart(
        "e:/nlp-for-disaster/exp3/results/model_comparison_metrics.csv",
        "e:/nlp-for-disaster/exp3/results/model_comparison_chart.png",
        "Experiment 3: 2-Agent Sequential Classification F1-Score Comparison"
    )
    
    # Cross architecture charts
    cross_csv = "e:/nlp-for-disaster/exp3/results/exp3_vs_other_comparison.csv"
    if os.path.exists(cross_csv):
        generate_cross_arch_chart(cross_csv, "e:/nlp-for-disaster/exp3/results/cross_architecture_comparison.png")
        generate_latency_comparison(cross_csv, "e:/nlp-for-disaster/exp3/results/latency_comparison.png")

if __name__ == '__main__':
    main()
