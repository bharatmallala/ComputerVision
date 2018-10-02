# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:25:34 2018

@author: bhara
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

#loading image
img1 = cv2.imread("test_MMS Trade Payables_____4M02KBGA_MMS Invoice (with PO - Normal).tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("test_MMS Trade Payables_____321Z21J_06YPTF2NB004G4F_MMS Invoice (with PO - High).tif", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("test_MMS Trade Payables_____321Z221_07DZ2J1VV003SEP_MMS Invoice (with PO - High).tif", cv2.IMREAD_GRAYSCALE)

#Non-local means denoising
new1 = cv2.fastNlMeansDenoising(img1,None,30,31,9)
new2 = cv2.fastNlMeansDenoising(img2,None,30,31,9)
new3 = cv2.fastNlMeansDenoising(img3,None,30,31,9)


k = np.ones((5,5), np.float32)/25
#k = np.ones((15,15), np.float32)/225

#2D filtering
smooth = cv2.filter2D(img1, -1,k)
smooth1 = cv2.filter2D(img2, -1,k)
smooth2 = cv2.filter2D(img3, -1,k)


#Average blurring
blur = cv2.blur(img1,(5,5))
blur1 = cv2.blur(img2,(5,5))
blur2 = cv2.blur(img3,(5,5))


#Gaussian Blurring
gausblur = cv2.GaussianBlur(img1,(5,5),0)
gausblur1 = cv2.GaussianBlur(img2,(5,5),0)
gausblur2 = cv2.GaussianBlur(img3,(5,5),0)

#Median Blurring 
median = cv2.medianBlur(img1,5)
median1 = cv2.medianBlur(img2,5)
median2 = cv2.medianBlur(img3,5)

#Bilateral Filtering
bilateral = cv2.bilateralFilter(img1,9,75,75)
bilateral1 = cv2.bilateralFilter(img2,9,75,75)
bilateral2 = cv2.bilateralFilter(img3,9,75,75)


# Checking results with OCR using tesseract
i = Image.fromarray(img1)
n = Image.fromarray(new1)
n1 = Image.fromarray(gausblur)
text = pytesseract.image_to_string(i,lang = 'eng')
text1 = pytesseract.image_to_string(n,lang = 'eng')
text2 = pytesseract.image_to_string(n1,lang = 'eng')



#plotting images
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(img1)
#plt.subplot(122),plt.imshow(gausblur)
plt.show()  


#output images
#cv2.imwrite('newimg3.tif', gausblur)
cv2.imwrite('newimg.tif',new1)
cv2.imwrite('newimg1.tif',new2)
cv2.imwrite('newimg2.tif',new3)