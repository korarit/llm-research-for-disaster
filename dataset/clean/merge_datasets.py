import os
import pandas as pd

def merge_datasets():
    # Define directories relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = script_dir
    
    # Define files and their corresponding source names
    files_mapping = {
        'dataset_facebook-comments-scraper_haiyai_water_flood_from_กองทัพบกทันกระแส.csv': 'กองทัพบกทันกระแส',
        'dataset_facebook-comments-scraper_haiyai_water_flood_from_พรรคประชาชน.csv': 'พรรคประชาชน',
        'dataset_facebook-comments-scraper_haiyai_water_flood_from_โลกใบใหญ่ของอองรี.csv': 'โลกใบใหญ่ของอองรี'
    }
    
    dfs = []
    
    print("Reading and marking cleaned datasets...")
    for filename, source in files_mapping.items():
        file_path = os.path.join(clean_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Cleaned file not found: {file_path}. Please run clean_dataset.py first.")
            continue
            
        try:
            # Read cleaned dataset
            df = pd.read_csv(file_path)
            # Add source name in the column 'marge' as requested
            df['marge'] = source
            dfs.append(df)
            print(f"  Read {filename} - {len(df)} rows from '{source}'")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            
    if not dfs:
        print("No datasets were successfully loaded. Exiting.")
        return
        
    # Merge all datasets
    merged_df = pd.concat(dfs, ignore_index=True)
    total_before = len(merged_df)
    print(f"\nTotal rows before deduplication: {total_before}")
    
    # Check for duplicates on the 'text' column
    duplicate_mask = merged_df.duplicated(subset=['text'], keep='first')
    duplicate_count = duplicate_mask.sum()
    
    # Print the duplicates count
    print(f"Number of duplicate texts found: {duplicate_count}")
    
    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
    total_after = len(merged_df)
    print(f"Total rows after removing duplicates: {total_after} (Removed {duplicate_count} rows)")
    
    # Re-assign sequential ID column starting from 1
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    # Re-order columns: id, text, marge
    merged_df = merged_df[['id', 'text', 'marge']]
    
    # Save the merged dataset
    output_path = os.path.join(clean_dir, 'merged_clean.csv')
    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully saved merged dataset to: {output_path}")
    except Exception as e:
        print(f"Error saving merged dataset: {e}")

if __name__ == '__main__':
    merge_datasets()
