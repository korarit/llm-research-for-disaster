import subprocess
import sys
import os

def main():
    print("Running Experiment 2E models in parallel...")
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    commands = [
        (["python", "-u", "e:/nlp-for-disaster/exp2E/run_typhoon.py"], "typhoon-v2.5"),
        (["python", "-u", "e:/nlp-for-disaster/exp2E/run_deepseek.py"], "deepseek-v4-flash"),
        (["python", "-u", "e:/nlp-for-disaster/exp2E/run_gemma.py"], "gemma-4")
    ]
    
    processes = []
    for cmd, name in commands:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        processes.append((p, name))
        
    active_processes = list(processes)
    while active_processes:
        for p, name in list(active_processes):
            output = p.stdout.readline()
            if output:
                print(f"[{name}] {output.strip()}")
            if p.poll() is not None:
                for line in p.stdout.read().splitlines():
                    print(f"[{name}] {line.strip()}")
                print(f"Finished {name} evaluation.")
                active_processes.remove((p, name))
                
    print("\nAll models finished execution. Evaluating Experiment 2E results...")
    subprocess.run(["python", "e:/nlp-for-disaster/exp2E/evaluate.py"], shell=True)
    print("Experiment 2E complete.")

if __name__ == '__main__':
    main()
