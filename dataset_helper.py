import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def main():
    dataset_dir = "e:/nlp-for-disaster/dataset/crisis-mmd"
    output_file = "e:/nlp-for-disaster/dataset/dataset_sample_1000.csv"
    
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
    required_cols = ['tweet_id', 'tweet_text', 'text_info', 'text_human', 'image_human', 'disaster_source']
    for col in required_cols:
        if col not in merged_df.columns:
            raise ValueError(f"Required column {col} not found in dataset!")
            
    df_all = merged_df[required_cols].copy()
    df_all = df_all.dropna(subset=['tweet_text', 'text_info', 'text_human', 'image_human'])
    df_all['tweet_text'] = df_all['tweet_text'].apply(lambda x: " ".join(str(x).split()))
    df_all = df_all[df_all['tweet_text'].str.strip() != ""]
    
    # Load previous 500 tweet_ids if file exists
    prev_sample_path = "e:/nlp-for-disaster/dataset/dataset_sample_500.csv"
    prev_ids = set()
    if os.path.exists(prev_sample_path):
        prev_df = pd.read_csv(prev_sample_path)
        prev_ids = set(prev_df['tweet_id'].tolist())
        print(f"Loaded {len(prev_ids)} previous sample tweet_ids for exclusion.")
        
    # Targets for the 1000 dataset (2x the old dataset counts)
    targets = {
        'other_relevant_information': 304,
        'not_humanitarian': 250,
        'rescue_volunteering_or_donation_effort': 190,
        'infrastructure_and_utility_damage': 90,
        'affected_individuals': 52,
        'injured_or_dead_people': 52,
        'vehicle_damage': 32,
        'missing_or_found_people': 30
    }
    
    sampled_buckets = {}
    
    for cat, target in targets.items():
        cat_df = df_all[df_all['text_human'] == cat]
        
        # Tier definitions
        t1 = cat_df[(cat_df['text_human'] == cat_df['image_human']) & (cat_df['tweet_text'].str.len() >= 70) & (~cat_df['tweet_id'].isin(prev_ids))]
        t2 = cat_df[(cat_df['text_human'] == cat_df['image_human']) & (cat_df['tweet_text'].str.len() >= 50) & (~cat_df['tweet_id'].isin(prev_ids))]
        t3 = cat_df[(cat_df['tweet_text'].str.len() >= 70) & (~cat_df['tweet_id'].isin(prev_ids))]
        t4 = cat_df[(cat_df['tweet_text'].str.len() >= 50) & (~cat_df['tweet_id'].isin(prev_ids))]
        t5 = cat_df[(cat_df['text_human'] == cat_df['image_human']) & (cat_df['tweet_text'].str.len() >= 70)]
        t6 = cat_df[(cat_df['text_human'] == cat_df['image_human']) & (cat_df['tweet_text'].str.len() >= 50)]
        t7 = cat_df[(cat_df['tweet_text'].str.len() >= 70)]
        t8 = cat_df[cat_df['tweet_text'].str.len() >= 50]
        t9 = cat_df[cat_df['tweet_text'].str.len() >= 30]

        tiers = [
            ('Tier 1 (Match + >=70 + Exclude)', t1),
            ('Tier 2 (Match + >=50 + Exclude)', t2),
            ('Tier 3 (No Match + >=70 + Exclude)', t3),
            ('Tier 4 (No Match + >=50 + Exclude)', t4),
            ('Tier 5 (Match + >=70 + Include)', t5),
            ('Tier 6 (Match + >=50 + Include)', t6),
            ('Tier 7 (No Match + >=70 + Include)', t7),
            ('Tier 8 (No Match + >=50 + Include)', t8),
            ('Tier 9 (No Match + >=30 + Include)', t9),
        ]
        
        selected = pd.DataFrame()
        for name, tier_df in tiers:
            if selected.empty:
                pool = tier_df
            else:
                pool = tier_df[~tier_df['tweet_id'].isin(selected['tweet_id'])]
            
            needed = target - len(selected)
            if needed <= 0:
                break
            if len(pool) >= needed:
                selected = pd.concat([selected, pool.sample(n=needed, random_state=42)])
                print(f"{cat}: Reached target {target} using {name} (added {needed} items)")
                break
            else:
                selected = pd.concat([selected, pool])
                print(f"{cat}: Added all {len(pool)} items from {name}")
                
        print(f"{cat} final selection size: {len(selected)}\n")
        sampled_buckets[cat] = selected

    sample_df = pd.concat(sampled_buckets.values(), ignore_index=True)
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
