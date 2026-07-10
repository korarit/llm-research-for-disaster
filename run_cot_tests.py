import subprocess
import sys
import os
import time

def main():
    experiments = ["exp1E-COT", "exp2E-COT", "exp3E-COT"]
    start_time = time.time()
    
    for exp in experiments:
        print(f"\n==================================================")
        print(f"STARTING {exp} WITH COT PROMPTS...")
        print(f"==================================================\n")
        
        # Clean results folder first to avoid confusion
        results_dir = f"e:/nlp-for-disaster/{exp}/results"
        if os.path.exists(results_dir):
            print(f"Cleaning existing results in {results_dir}...")
            for file in os.listdir(results_dir):
                file_path = os.path.join(results_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
        
        exp_dir = f"e:/nlp-for-disaster/{exp}"
        p = subprocess.run([sys.executable, "-u", "run_all.py"], cwd=exp_dir)
        
        print(f"\n==================================================")
        print(f"FINISHED {exp} WITH RETURN CODE {p.returncode}")
        print(f"==================================================\n")
        
    elapsed = time.time() - start_time
    print(f"All COT tests completed in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
