import subprocess
import sys
import os
import threading
from queue import Queue, Empty
import time

def enqueue_output(out, queue, name):
    for line in iter(out.readline, ''):
        queue.put((name, line))
    out.close()

def main():
    print("Running Experiment 3F models in parallel...")
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    commands = [
        ([sys.executable, "-u", "e:/nlp-for-disaster/exp3F/run_typhoon.py"], "typhoon-v2.5"),
        ([sys.executable, "-u", "e:/nlp-for-disaster/exp3F/run_deepseek.py"], "deepseek-v4-flash"),
        ([sys.executable, "-u", "e:/nlp-for-disaster/exp3F/run_gemma.py"], "gemma-4")
    ]
    
    processes = []
    q = Queue()
    
    for cmd, name in commands:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        t = threading.Thread(target=enqueue_output, args=(p.stdout, q, name))
        t.daemon = True
        t.start()
        processes.append((p, name, t))
        
    active_processes = list(processes)
    while active_processes:
        try:
            while True:
                name, line = q.get_nowait()
                print(f"[{name}] {line.strip()}")
                q.task_done()
        except Empty:
            pass
            
        for item in list(active_processes):
            p, name, t = item
            if p.poll() is not None:
                t.join(timeout=1.0)
                print(f"Finished {name} evaluation.")
                active_processes.remove(item)
                
        time.sleep(0.1)
        
    print("\nAll models finished execution. Evaluating Experiment 3F results...")
    subprocess.run([sys.executable, "e:/nlp-for-disaster/exp3F/evaluate.py"])
    print("Experiment 3F complete.")

if __name__ == '__main__':
    main()
