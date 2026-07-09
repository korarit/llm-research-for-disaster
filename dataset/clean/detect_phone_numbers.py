import os
import re
import pandas as pd
from collections import Counter

def detect_and_export_phones():
    # Define paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'merged_clean.csv')
    export_path = os.path.join(script_dir, 'detected_phones.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
        
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Regex pattern matching Thai mobile phone numbers:
    # Starts with 06, 08, 09 or +66/66 (optionally separated) followed by 8 digits
    # Interspersed with separators (spaces, hyphens, dots, slashes)
    pattern = r'(?<!\d)(?:0[689]|\+?66[-.\s/]*[689])[-.\s/]*(?:\d[-.\s/]*){8}(?!\d)'
    
    def clean_and_normalize(match):
        digits = re.sub(r'\D', '', match)
        if digits.startswith('66'):
            digits = '0' + digits[2:]
        return digits

    def find_phones(text):
        if not isinstance(text, str):
            return []
        matches = re.findall(pattern, text)
        return [clean_and_normalize(m) for m in matches]

    print("Extracting phone numbers...")
    all_normalized_phones = []
    detect_phone_flags = []
    
    for idx, row in df.iterrows():
        phones = find_phones(row['text'])
        if phones:
            detect_phone_flags.append(True)
            all_normalized_phones.extend(phones)
        else:
            detect_phone_flags.append(False)
            
    # Add column to the dataframe
    df['detect_phone'] = detect_phone_flags
    
    # Save the updated merged_clean.csv
    print(f"Saving updated dataset with 'detect_phone' column to: {input_path}")
    df.to_csv(input_path, index=False, encoding='utf-8-sig')
    
    # Count occurrences of each phone number
    phone_counts = Counter(all_normalized_phones)
    
    # Create DataFrame for exporting
    export_df = pd.DataFrame(phone_counts.items(), columns=['phone_number', 'count'])
    # Sort by count descending, then by phone number ascending
    export_df = export_df.sort_values(by=['count', 'phone_number'], ascending=[False, True])
    
    # Save the exported CSV
    print(f"Exporting detected phone numbers and counts to: {export_path}")
    export_df.to_csv(export_path, index=False, encoding='utf-8-sig')
    
    # Display some stats
    total_detected_rows = sum(detect_phone_flags)
    print("\nSummary Stats:")
    print(f"  Total rows processed: {len(df)}")
    print(f"  Rows with phone numbers: {total_detected_rows} ({total_detected_rows / len(df) * 100:.2f}%)")
    print(f"  Unique phone numbers detected: {len(export_df)}")
    print(f"  Total phone numbers extracted: {len(all_normalized_phones)}")
    
    if not export_df.empty:
        print("\nTop 10 most frequent phone numbers:")
        print(export_df.head(10).to_string(index=False))

if __name__ == '__main__':
    detect_and_export_phones()
