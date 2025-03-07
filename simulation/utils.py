import os

def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)