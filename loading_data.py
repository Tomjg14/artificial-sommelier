import os, random
import ntpath
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import math
from pprint import pprint
import json

from pandas.io.json import json_normalize

file_path = "data/winemag-data-130k-v2.json"          

with open(file_path) as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')


