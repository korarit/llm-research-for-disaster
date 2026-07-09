import os

print("Searching E:\\nlp-for-disaster recursively...")
found = False
for root, dirs, files in os.walk("E:\\nlp-for-disaster"):
    for f in files:
        if f.endswith('.tsv') or f.endswith('.csv') or 'crisis' in f.lower() or 'thai_500' in f.lower():
            print("File:", os.path.join(root, f))
            found = True
    for d in dirs:
        if 'crisis' in d.lower() or 'data' in d.lower():
            print("Dir:", os.path.join(root, d))
            found = True

if not found:
    print("No matching files or directories found in the workspace.")
