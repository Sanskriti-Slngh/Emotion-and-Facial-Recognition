import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import pickle
import face_recognition

file_name = []

for i in range(1,2):
    file = 'data/att_faces/s3/'+ str(i) + '.pgm'
    image = cv2.imread(file)
    plt.imshow(image)
    plt.show()
    image_hflip = cv2.flip(image,-1)
    plt.imshow(image_hflip)
    plt.show()
