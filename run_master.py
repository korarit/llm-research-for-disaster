import os
import shutil
import subprocess
import time
import sys
import threading
from queue import Queue, Empty

# List of all experiments to run
experiments = [
    "exp1", "exp1E", "exp1F",
    "exp2", "exp2E", "exp2F",
    "exp3", "exp3E", "exp3F"
]

def clean_results():
    print("Cleaning up old results...")
    for exp in experiments:
        results_dir = f"e:/nlp-for-disaster/{exp}/results"
        if os.path.exists(results_dir):
            print(f"Deleting contents of {results_dir}")
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"Failed to delete {item_path}: {e}")
        else:
            print(f"Directory {results_dir} does not exist.")

def enqueue_output(out, queue, exp):
    for line in iter(out.readline, ''):
        queue.put((exp, line))
    out.close()

def run_experiments():
    from dotenv import load_dotenv
    load_dotenv()
    
    key1 = os.getenv("TYPHOON_API_KEY")
    key2 = os.getenv("TYPHOON_SECOND_API_KEY")
    
    if not key1 or not key2:
        print("Error: Both TYPHOON_API_KEY and TYPHOON_SECOND_API_KEY must be set in .env")
        sys.exit(1)
        
    print(f"Loaded keys: {key1[:10]}... and {key2[:10]}...")
    
    queue = list(experiments)
    active_jobs = {} # slot_index: { 'process': p, 'exp': exp, 'thread': t, 'log_file': f }
    q = Queue()
    
    keys = [key1, key2]
    key_index = 0
    
    while queue or active_jobs:
        # Fill empty slot (allow max 1 active job at a time)
        if len(active_jobs) < 1 and queue:
            exp = queue.pop(0)
            selected_key = keys[key_index]
            key_index = (key_index + 1) % len(keys)
            
            print(f"\n[Master] Starting {exp} using key: {selected_key[:10]}...")
            
            env = os.environ.copy()
            env["TYPHOON_API_KEY"] = selected_key
            env["PYTHONUNBUFFERED"] = "1"
            
            exp_dir = f"e:/nlp-for-disaster/{exp}"
            p = subprocess.Popen(
                ["python", "-u", f"{exp}/run_all.py"],
                cwd="e:/nlp-for-disaster",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                shell=True
            )
            
            # Start a thread to read output of this subprocess
            t = threading.Thread(target=enqueue_output, args=(p.stdout, q, exp))
            t.daemon = True
            t.start()
            
            # We use 0 as the slot index for the single active job
            active_jobs[0] = {
                'process': p,
                'exp': exp,
                'thread': t,
                'log_file': open(f"e:/nlp-for-disaster/{exp}_run.log", "w", encoding="utf-8")
            }
                
        # Read all available output from the queue
        try:
            while True:
                exp, line = q.get_nowait()
                cleaned_line = f"[{exp}] {line.strip()}"
                print(cleaned_line)
                
                # Find log file for this exp
                for slot, job in active_jobs.items():
                    if job['exp'] == exp:
                        job['log_file'].write(line)
                        job['log_file'].flush()
                        break
                q.task_done()
        except Empty:
            pass
            
        # Check if any processes have finished
        for slot in list(active_jobs.keys()):
            job = active_jobs[slot]
            p = job['process']
            exp = job['exp']
            
            if p.poll() is not None:
                # Wait for reader thread to finish
                job['thread'].join(timeout=1.0)
                
                # Consume any remaining items in queue for this exp
                print(f"[Master] Job {exp} finished with return code {p.returncode}")
                job['log_file'].close()
                del active_jobs[slot]
                
        time.sleep(0.1)

if __name__ == '__main__':
    clean_results()
    run_experiments()
