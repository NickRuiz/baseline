import json
import pymongo
import pandas as pd
import os
__all__ = ["log2json"]


def log2json(log_file):
    s = []
    with open(log_file) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s
