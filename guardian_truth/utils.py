import random, time, json
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def now_ms():
    return time.perf_counter()*1000
