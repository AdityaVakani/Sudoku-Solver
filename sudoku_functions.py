import cv2
import PIL
import numpy as np
import operator
from fastai.vision import *


path=Path('C:\\Users\\adivf\Desktop\\Sudoku Related\\sudoku_dset')
learn = load_learner(path)

def pre_processing(img,dilate=False):
    proc=cv2.GaussianBlur(img,(3,3),0)
    proc=cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    if dilate != False:
        kernel=np.array((2,2),np.float64)
        proc = cv2.dilate(proc,kernel)
    return proc



def find_corners(img):
    contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    polygon=contours[0]


    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left,_ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))


    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



def distance(pt2,pt1):
    x=pt2[0]-pt1[0]
    y=pt2[1]-pt2[1]
    return np.sqrt(x**2 +y**2)



#fix it
def cropimage(img,crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        distance(bottom_right, top_right),
        distance(top_left, bottom_left),
        distance(bottom_right, bottom_left),
        distance(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img):
    squares=[]
    side=img.shape[:1]
    side=side[0]/9
    for i in range(9):
        for j in range(9):
            p1=(i* side, j* side)
            p2=((i+1)* side,(j+1)* side)
            squares.append((p1,p2))
    return squares

def display_rects(img, rects, colour=(0, 0, 255)):
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    cv2.imshow('rect',img)
    return img



def get_blocks(img,rects):
    for i in range(9):
        for j in range(9):
            bx1=rects[9*i+j]
            box1 = img[int(bx1[0][0]):int(bx1[1][0]), int(bx1[0][1]):int(bx1[1][1])]
            box1=cv2.resize(box1,(128,128))
            cv2.imshow('box',box1)
            k = cv2.waitKey(0) & 0xFF
            if k==27:
                cv2.destroyWindow()
                break
            cv2.destroyWindow('box')
    return



def show_countours(img):
    contours, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    borders = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 2)
    cv2.imshow('bord', borders)


def get_array(img,rects,zero_pos):
    array=[[0 for i in range(9)] for i in range(9)]
    for i in range(9):
        for j in range(9):
            bx1=rects[9*i+j]
            box1 = img[int(bx1[0][0]):int(bx1[1][0]), int(bx1[0][1]):int(bx1[1][1])]
            box1=pre_processing(box1)
            box1=cv2.resize(box1,(128,128))
            box1 = Image(pil2tensor(box1, np.float32).div_(255))
            pred_class, pred_idx, outputs = learn.predict(box1)
            array[i][j]=int(pred_class)
            if int(pred_class)==0:
                zero_pos.append([i,j])
    return array
def write_image(img,array,zero_pos,rects):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for pos in zero_pos:
        r=pos[0]
        c=pos[1]
        text = str(array[r][c])
        bx1 = rects[9 * r + c]
        centre = (int((int(bx1[0][1])+int(bx1[1][1]))/2),int((int(bx1[0][0])+int(bx1[1][0]))/2))
        cv2.putText(img,text,centre,font,1,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow('sol',img)
    cv2.waitKey()
    cv2.destroyWindow('sol')
    return
def printgrid(arr):
    for i in range(9):
        print(arr[i])


def emptypos(arr, l):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


def checkrow(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


def checkcol(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


def checkbox(arr, row, col, num):
    row1 = row - (row % 3)
    col1 = col - (col % 3)
    for i in range(3):
        for j in range(3):
            if arr[row1 + i][col1 + j] == num:
                return True
    return False


def checkifsafe(arr, row, col, num):
    return not checkrow(arr, row, num) and not checkcol(arr, col, num) and not checkbox(arr, row, col, num)


def solvesudoku(arr):
    l = [0, 0]
    if (not emptypos(arr, l)):
        return True
    row = l[0]
    col = l[1]

    for num in range(1, 10):
        if checkifsafe(arr, row, col, num):
            arr[row][col] = num
            if solvesudoku(arr):
                return True
            arr[row][col] = 0

    return False

