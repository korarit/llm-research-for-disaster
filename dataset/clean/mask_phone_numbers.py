import os
import re
import pandas as pd

def mask_phone_numbers_in_all_files():
    # Define directories relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # We will look for CSV files in both dataset/ and dataset/clean/
    target_dirs = [parent_dir, script_dir]
    csv_files = []
    
    for d in target_dirs:
        for f in os.listdir(d):
            if f.endswith('.csv'):
                csv_files.append(os.path.join(d, f))
                
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Regex pattern matching Thai mobile phone numbers:
    pattern = r'(?<!\d)(?:0[689]|\+?66[-.\s/]*[689])[-.\s/]*(?:\d[-.\s/]*){8}(?!\d)'
    
    def mask_phone_match(match_str):
        chars = list(match_str)
        digits_masked = 0
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].isdigit():
                chars[i] = '*'
                digits_masked += 1
                if digits_masked == 4:
                    break
        return "".join(chars)

    def mask_value(val):
        if pd.isna(val) or not isinstance(val, str):
            return val
        return re.sub(pattern, lambda m: mask_phone_match(m.group(0)), val)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))
        print(f"\nMasking phone numbers in: {dir_name}/{filename}")
        
        try:
            # Read all columns as string to preserve exact text format (prevent number formatting changes)
            df = pd.read_csv(file_path, dtype=str)
            
            # Count elements before masking to check if we made changes
            # Apply masking to all elements in the dataframe
            # We use apply map to safely process cell by cell
            df_masked = df.apply(lambda col: col.map(mask_value))
            
            # Save back in-place
            df_masked.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  Successfully processed and saved.")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == '__main__':
    mask_phone_numbers_in_all_files()
