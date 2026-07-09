import subprocess
import sys
import os

def main():
    print("Running Experiment 2 models in parallel...")
    processes = [
        subprocess.Popen([sys.executable, "-u", "exp2/run_deepseek.py"]),
        subprocess.Popen([sys.executable, "-u", "exp2/run_typhoon.py"]),
        subprocess.Popen([sys.executable, "-u", "exp2/run_gemma.py"])
    ]
    
    # Wait for all to finish
    for p in processes:
        p.wait()
        
    print("\nAll models finished execution. Evaluating Experiment 2 results...")
    subprocess.run([sys.executable, "exp2/evaluate.py"])
    print("Experiment 2 complete.")

if __name__ == '__main__':
    main()
