from os.path import dirname, join, basename
import sys
from glob import glob
import cv2
from cv2 import ml
import numpy as np


bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:int(x_pixel/2),:int(y_pixel/2)], bins[int(x_pixel/2):,:int(y_pixel/2)], bins[:int(x_pixel/2),int(y_pixel/2):], bins[int(x_pixel/2):,int(y_pixel/2):]
    mag_cells = mag[:int(x_pixel/2),:int(y_pixel/2)], mag[int(x_pixel/2):,:int(y_pixel/2)], mag[:int(x_pixel/2),int(y_pixel/2):], mag[int(x_pixel/2):,int(y_pixel/2):]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
#    print hist.shape
#    print type(hist)
    return hist



samples = []
labels = []    

# Get positive samples
for filename in glob(join('./cat', '*.jpg')):
    img = cv2.imread(filename, 1)
    hist = hog(img)
    samples.append(hist)
    labels.append(1)

# Get negative samples
for filename in glob(join('./other', '*.jpg')):
    img = cv2.imread(filename, 1)
    hist = hog(img)
    samples.append(hist)
    labels.append(-1)

# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)


# Create SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR) # cv2.ml.SVM_LINEAR
svm.setGamma(5.383)
svm.setC(2.67)


# Train
print ("Samples : ",samples.shape)
print ("labels : ",labels.shape)
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')