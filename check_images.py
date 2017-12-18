'''
Script to check whether images are corrupted. Without an argument, checks `data/wikiart/`. Otherwise checks 
the `data/<dataset-name>`
Usage: 
    python check_images.py <dataset-name>
    
    <dataset-name> : `data/<dataset-name>`
'''
from utils import *
from glob import glob
import sys

if len(sys.argv) == 1:
    test_images(glob("./data/wikiart/*/*.jpg"))
else:
    test_images(glob("./data/" + str(sys.argv[1]) + "/*/*.jpg"))

