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



svm = cv2.ml.SVM_create()
svm = cv2.ml.SVM_load('svm_data.dat')



test_temp=[]
for fn in glob(join('./predict', '*.jpg')):
    img=cv2.imread(fn,0) #参数加0，只读取黑白数据，去掉0，就是彩色读取。
    test_temp.append(hog(img))


for item in test_temp:
    reshape_item = np.float32(item).reshape(-1,bin_n*4)
    result = svm.predict(reshape_item)
    #print (result)
    if (result[1][0][0] >0):
        print ("cat")
    else:
        print ("nocat")
