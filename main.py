import cv2
import numpy as np
from pyocr.pyocr import PyOCR

model = PyOCR()
image_path = "test.png"
image_ndarray = np.array(cv2.imread(image_path))
result = model.ocr(image_ndarray)
print(result)
