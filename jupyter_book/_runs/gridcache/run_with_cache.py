"""Run a book script with the grid cache active and a heartbeat line every 60 s.
Usage (from C2/): python ../_runs/gridcache/run_with_cache.py <script.py> [args...]"""
import os, runpy, subprocess, sys, threading, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_grid_cache  # noqa: E402,F401  (patches femm_solver.MeshInterpolator)


def heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                  '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        except Exception:
            gpu = 'n/a'
        rss = int(open(f'/proc/{os.getpid()}/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30
        load = os.getloadavg()[0]
        print(f'[heartbeat {time.strftime("%H:%M:%S")}] elapsed {(time.time()-t0)/60:.1f} min | GPU {gpu} | '
              f'RSS {rss:.2f} GB | load {load:.1f} | cache {mesh_grid_cache.STATS}', flush=True)


threading.Thread(target=heartbeat, daemon=True).start()
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
