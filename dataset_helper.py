import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def main():
    dataset_dir = "e:/nlp-for-disaster/dataset/crisis-mmd"
    output_file = "e:/nlp-for-disaster/dataset/dataset_sample_500.csv"
    
    print(f"Reading TSV files from {dataset_dir}...")
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.tsv')]
    
    all_dfs = []
    for f in files:
        file_path = os.path.join(dataset_dir, f)
        disaster_name = f.replace("_final_data.tsv", "")
        df = pd.read_csv(file_path, sep='\t')
        
        # Add disaster source context
        df['disaster_source'] = disaster_name
        all_dfs.append(df)
        
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Filter columns and clean
    required_cols = ['tweet_id', 'tweet_text', 'text_info', 'text_human', 'disaster_source']
    for col in required_cols:
        if col not in merged_df.columns:
            raise ValueError(f"Required column {col} not found in dataset!")
            
    df_clean = merged_df[required_cols].copy()
    df_clean = df_clean.dropna(subset=['tweet_text', 'text_info', 'text_human'])
    df_clean['tweet_text'] = df_clean['tweet_text'].apply(lambda x: " ".join(str(x).split()))
    df_clean = df_clean[df_clean['tweet_text'].str.strip() != ""]
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    # Show distribution of text_human
    print("\nOriginal text_human distribution:")
    print(df_clean['text_human'].value_counts())
    
    target_sample_size = 500
    counts = df_clean['text_human'].value_counts()
    categories = counts.index.tolist()
    
    # Allocate initial minimum samples of 15 (or all available if less)
    allocated = {}
    for cat in categories:
        allocated[cat] = min(counts[cat], 15)
        
    remaining = target_sample_size - sum(allocated.values())
    
    # Distribute the rest proportionally to the remaining unsampled parts
    initial_remaining = remaining
    unsampled_counts = {cat: counts[cat] - allocated[cat] for cat in categories}
    total_unsampled = sum(unsampled_counts.values())
    
    if total_unsampled > 0:
        for cat in categories:
            add = int(round((unsampled_counts[cat] / total_unsampled) * initial_remaining))
            add = min(unsampled_counts[cat], add)
            allocated[cat] += add
            
    # Adjust to get exactly target_sample_size
    current_total = sum(allocated.values())
    diff = target_sample_size - current_total
    
    if diff > 0:
        for cat in categories:
            space = counts[cat] - allocated[cat]
            add = min(space, diff)
            allocated[cat] += add
            diff -= add
            if diff == 0:
                break
    elif diff < 0:
        for cat in reversed(categories):
            excess = allocated[cat] - 15
            sub = min(excess, -diff)
            allocated[cat] -= sub
            diff += sub
            if diff == 0:
                break

    # Perform the sampling
    sampled_dfs = []
    for cat, size in allocated.items():
        cat_df = df_clean[df_clean['text_human'] == cat]
        sampled_dfs.append(cat_df.sample(n=size, random_state=42))
        
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle the final sample
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nSampled dataset shape: {sample_df.shape}")
    print("\nSampled text_human distribution:")
    print(sample_df['text_human'].value_counts())
    print(f"Total samples: {len(sample_df)}")
    
    # Save the sample CSV
    sample_df.to_csv(output_file, index=False)
    print(f"\nSaved sample to {output_file}")

if __name__ == '__main__':
    main()
