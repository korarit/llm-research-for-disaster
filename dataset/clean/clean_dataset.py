import os
import pandas as pd
import glob

def clean_datasets():
    # Define directories relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.dirname(script_dir)  # parent dir: dataset/
    target_dir = script_dir                    # target dir: dataset/clean/
    
    # Ensure clean directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Locate all CSV files in the source directory (excluding the clean subdirectory)
    csv_files = [f for f in glob.glob(os.path.join(source_dir, '*.csv')) if os.path.isfile(f)]
    
    print(f"Found {len(csv_files)} raw CSV files to clean.")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # Read CSV file (using utf-8-sig or letting pandas detect encoding)
            df = pd.read_csv(file_path, low_memory=False)
            
            if 'text' not in df.columns:
                print(f"Warning: 'text' column not found in {filename}. Skipping.")
                continue
            
            # Select only 'text' column
            cleaned_df = df[['text']].copy()
            
            # Remove rows where 'text' is null or whitespace-only
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['text'].notnull()]
            cleaned_df = cleaned_df[cleaned_df['text'].astype(str).str.strip() != '']
            final_count = len(cleaned_df)
            
            # Assign sequential ID starting from 1
            cleaned_df.insert(0, 'id', range(1, len(cleaned_df) + 1))
            
            # Save to clean directory
            target_path = os.path.join(target_dir, filename)
            cleaned_df.to_csv(target_path, index=False, encoding='utf-8-sig')
            
            print(f"  Successfully cleaned: {initial_count} rows -> {final_count} rows")
            print(f"  Saved to: {target_path}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == '__main__':
    clean_datasets()
