import cv2
import numpy as np
import glob
import datetime
import os


def walk():
    folder = "in_images"
    fol = []
    for cd, dir, files in os.walk(folder):
        if dir != []:
            fol = dir
    print(f"fol : {fol}")
    return fol
    

def adjust(img, alpha, beta):
    # 積和演算を行う。
    dst = alpha * img + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)


fol = walk()
for f in fol:
    n = 0
    files = glob.glob(fr"./in_images/{f}/*.pgm", recursive=True) 

    for file in files:
        # 画像を読み込む。
        src = cv2.imread(file)
        # コントラスト、明るさを変更する。
        alpha=1
        beta=50.0
        dst = adjust(src, alpha, beta)
        img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'./in_images/{f}/brt{beta}_img_{n}.pgm',img_gray)
        n = n + 1
