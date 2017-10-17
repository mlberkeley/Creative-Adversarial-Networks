from utils import *
from glob import glob
import sys

if len(sys.argv) == 1:
    test_images(glob("./data/wikiart/*/*.jpg"))
else:
    test_images(glob("./data/" + str(sys.argv[1]) + "/*/*.jpg"))

