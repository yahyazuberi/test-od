import numpy as np
import cv2 as cv
import re
import pytesseract
from PIL import Image
import os
import shutil
import json


def detect(a):
    nic=[]
    date=[]
    
    pilImage = Image.fromarray(a)
    pilImage.save('15.png')    
    image = cv.imread("15.png")
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    text = []
    for i in range(10,200,10):
        th, threshed = cv.threshold(gray, i, 250, cv.THRESH_TRUNC)
        #cv.imwrite("filename"+str(i)+".png", threshed)
        st = pytesseract.image_to_string(threshed)
        text.append(st)
        cnic = re.search('([0-9]{5}(\s|)-(\s|)[0-9]{7}(\s|)-(\s|)[0-9])', st)
        date_issue = re.findall('([0-9]{2}.[0-9]{2}.20[0-9]{2})', st)
        if cnic:
           # print(i)
            #print(cnic.group())
            nic.append(cnic.group())
        if len(date_issue) > 0:
            #print(i)
            date.append((date_issue))
    #print(nic)
    print(max(set(nic), key = nic.count))
    #print(max(set(date), key = date.count))
    os.remove("15.png")






