import subprocess
import time
import os
import sys

# PATHS
python_path = "/home/exouser/runs/pinn-point/repo/.venv-netgen/bin/python"
repo_path = "/home/exouser/runs/pinn-point/repo"
meta_path = "/home/exouser/.remote_opps"
remote_entry = os.path.join(repo_path, ".remote_opps/remote_entry.py")

flavors = ["baseline", "narrow", "sharp", "narrow_sharp"]

log_dir = os.path.join(meta_path, "logs")
os.makedirs(log_dir, exist_ok=True)

for flavor in flavors:
    print(f"=== Starting Flavor: {flavor} ===")
    processes = []
    for i in range(4):
        start_seed = 101 + i * 5
        log_file = os.path.join(log_dir, f"seq_{flavor}_part{i}.log")
        spec_file = os.path.join(meta_path, f"specs/poisson_20seed_{flavor}.json")
        
        cmd = [
            python_path, remote_entry,
            "--spec", spec_file,
            "--start_seed", str(start_seed),
            "--num_seeds", "5"
        ]
        try:
            f = open(log_file, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=repo_path)
            processes.append((p, f))
            print(f"  Launched seeds {start_seed}-{start_seed+4} (PID: {p.pid})")
        except Exception as e:
            print(f"  FAILED to launch {start_seed}: {e}")
            if 'f' in locals(): f.close()
    
    if not processes:
        continue

    print(f"Waiting for {flavor} to complete...")
    for p, f in processes:
        p.wait()
        f.close()
    print(f"=== Completed Flavor: {flavor} ===")
