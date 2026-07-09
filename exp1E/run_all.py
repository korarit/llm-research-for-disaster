import subprocess
import sys
import os

def run_script(cmd, name):
    print(f"Starting {name}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[{name}] {output.strip()}")
    rc = process.poll()
    print(f"Finished {name} with code {rc}.")
    return rc

def main():
    print("Running Experiment 1E models in parallel...")
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    # We will run them concurrently.
    # To run concurrently in Python, we can spawn subprocess.Popen and wait for them.
    commands = [
        (["python", "-u", "e:/nlp-for-disaster/exp1E/run_typhoon.py"], "typhoon-v2.5"),
        (["python", "-u", "e:/nlp-for-disaster/exp1E/run_deepseek.py"], "deepseek-v4-flash"),
        (["python", "-u", "e:/nlp-for-disaster/exp1E/run_gemma.py"], "gemma-4")
    ]
    
    processes = []
    for cmd, name in commands:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        processes.append((p, name))
        
    active_processes = list(processes)
    while active_processes:
        for p, name in list(active_processes):
            # Check if there is output
            output = p.stdout.readline()
            if output:
                print(f"[{name}] {output.strip()}")
            # Check if process finished
            if p.poll() is not None:
                # Read remaining output
                for line in p.stdout.read().splitlines():
                    print(f"[{name}] {line.strip()}")
                print(f"Finished {name} evaluation.")
                active_processes.remove((p, name))
                
    print("\nAll models finished execution. Evaluating Experiment 1E results...")
    subprocess.run(["python", "e:/nlp-for-disaster/exp1E/evaluate.py"], shell=True)
    print("Experiment 1E complete.")

if __name__ == '__main__':
    main()
