
import sudoku_functions as sf
import cv2
import PIL
import numpy as np
import operator
from fastai.vision import *






img=cv2.imread('sudoku1t.jpg',0)
img=cv2.resize(img,(480,640))
cv2.imshow('img',img)
cv2.waitKey()

proc= sf.pre_processing(img,True)
cv2.imshow('proc',proc)
cv2.waitKey()
corners=sf.find_corners(proc)
cropped=sf.cropimage(img,corners)
crop_proc=sf.pre_processing(cropped)
cv2.imshow('cropped',crop_proc)
cv2.waitKey()
sqrs=sf.infer_grid(cropped)


zero_pos=[]
array=sf.get_array(cropped,sqrs,zero_pos)
#print(zero_pos)
sf.solvesudoku(array)
sf.printgrid(array)
sf.write_image(cropped.copy(),array,zero_pos,sqrs)


cv2.destroyAllWindows()


