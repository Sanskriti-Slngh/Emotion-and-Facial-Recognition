import cv2
import os
import re
import bz2
from matplotlib import pyplot as plt

path = 'D:/Datasets/data/emotions/'

for (dirname, dirpath, files) in os.walk(path):
    print (dirname, dirpath, len(files))
    emo = (dirname.split("/")[-1])
    emo = emo.title()
    for file in files:
        print (dirname + "/" + file, emo)
