import numpy as np
import cv2 as cv
import re
import pytesseract
from PIL import Image
import os
import shutil
import json

def detectYolo(h):
    
    h=json.loads(h)
    print("++++++++++++++++>>>>>>.")
    pilImage = Image.fromarray(h)
    pilImage.save('images/sc1.png')
    return "saved"
    