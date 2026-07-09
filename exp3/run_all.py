import subprocess
import sys
import os

def main():
    print("Running Experiment 3 models in parallel...")
    processes = [
        subprocess.Popen([sys.executable, "-u", "exp3/run_deepseek.py"]),
        subprocess.Popen([sys.executable, "-u", "exp3/run_typhoon.py"]),
        subprocess.Popen([sys.executable, "-u", "exp3/run_gemma.py"])
    ]
    
    # Wait for all to finish
    for p in processes:
        p.wait()
        
    print("\nAll models finished execution. Evaluating Experiment 3 results...")
    subprocess.run([sys.executable, "exp3/evaluate.py"])
    print("Experiment 3 complete.")

if __name__ == '__main__':
    main()
