import numpy as np
from pyaml_env import parse_config

def get_config(fpath="config.yml"):
    config = parse_config(path=fpath)
    return config

def build_name(prefix, **kwargs):
    string = prefix + "_"
    for k, v in kwargs.items():
        string += f"{k}-{v}_"
    return string[:-1] # Remove trailing underscore

def name_to_dict(expname):
    dct = dict()

    for kv in expname.split("_"):
        try:
            kv = kv.split("-")
        except ValueError:
            print(f"Skipping {kv}")
        dct[kv[0]] = "-".join(kv[1:])

    return dct

def project(segment):
    a, b = np.min(segment), np.max(segment)
    projected = 2 * (segment - a) / (b - a) - 1
    return projected
