import cv2
import numpy as np
import os
import sudoku_functions as sf

dest='C:\\Users\\adivf\\Desktop\\sudoku_dset\\Training\\'
name_count=[0 for i in range(0,10)]

def get_blocks(img,rects):
    for i in range(9):
        for j in range(9):
            bx1=rects[9*i+j]
            box1 = img[int(bx1[0][0]):int(bx1[1][0]), int(bx1[0][1]):int(bx1[1][1])]
            box1=cv2.resize(box1,(128,128))
            cv2.imshow('box',box1)
            k = cv2.waitKey(500) & 0xFF
            folder=input('enter number')
            img_dest=dest+folder+'\\'

            img_num=name_count[int(folder)]
            img_name=img_dest+str(img_num)+'.jpg'
            #print(img_name)
            cv2.imwrite(img_name,box1)
            name_count[int(folder)]+=1

            if k==27:
                cv2.destroyWindow('box')
                break
            cv2.destroyWindow('box')
    cv2.destroyAllWindows()

    return



for index in range(1,21):

    ind=str(index)
    source='C:\\Users\\adivf\Desktop\\sudoku images\\' + 'sudoku'+ind+'.jpg'
    img = cv2.imread(source,0)
    img=cv2.resize(img,(480,640))
    proc= sf.pre_processing(img,True)
    sf.show_countours(proc)

    corners=sf.find_corners(proc)
    cropped=sf.cropimage(img,corners)
    crop_proc=sf.pre_processing(cropped)
    sqrs=sf.infer_grid(cropped)

    get_blocks(crop_proc,sqrs)

    cv2.imshow('img',cropped)
    #cv2.waitKey(0)
cv2.destroyAllWindows()