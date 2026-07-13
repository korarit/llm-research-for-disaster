import subprocess
import sys
import os

def main():
    print("==================================================")
    print("STARTING ALL THAI EXPERIMENTS (1TH, 2TH, 3TH)")
    print("==================================================")
    
    experiments = ["exp1th", "exp2th", "exp3th"]
    
    for exp in experiments:
        print(f"\n>>> Running {exp}...")
        script_path = f"e:/nlp-for-disaster/{exp}/run_all.py"
        if os.path.exists(script_path):
            result = subprocess.run([sys.executable, script_path])
            if result.returncode != 0:
                print(f"Error: {exp} failed with return code {result.returncode}")
            else:
                print(f"Completed {exp} successfully.")
        else:
            print(f"Error: Script not found at {script_path}")
            
    print("\n==================================================")
    print("ALL THAI EXPERIMENTS COMPLETED.")
    print("==================================================")

if __name__ == '__main__':
    main()
