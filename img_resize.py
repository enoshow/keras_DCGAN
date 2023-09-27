import os
from PIL import Image
import glob
import cv2
import time

def walk():
    folder = "base_images"
    fol = []
    for cd, dir, files in os.walk(folder):
        if dir != []:
            fol = dir
    print(f"fol : {fol}")
    return fol

fol = walk()
x = 64

for f in fol:      
    files = glob.glob(fr"./base_images/{f}/*.pgm", recursive=True) 

    if os.path.isdir(f"./pr_images/{f}_{x}"):
        print(f"'./pr_images/{f}_{x}' : exist")
    else:
        os.mkdir(f"./pr_images/{f}_{x}")
    
    n = 1
    for file in files:
        im = Image.open(file)
        img = im.crop((x, x, 512-x, 512-x))
        img.save(f"./pr_images/{f}_{x}/pr_img_{x}_{n}.pgm", quality=95)
        n += 1

    time.sleep(0.01)

    n = 1
    files2 = glob.glob(fr"./pr_images/{f}_{x}/*.pgm", recursive=True)

    if os.path.isdir(f"./in_images/{f}"):
        print(f"'./in_images/{f}' : exist")
    else:
        os.mkdir(f"./in_images/{f}")

    for file2 in files2:
        img = cv2.imread(file2)
        img_512 = cv2.resize(img, (512, 512))
        img_gray = cv2.cvtColor(img_512, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./in_images/{f}/pr2_img_{x}_{n}.pgm", img_gray)
        n += 1
    
print("end")
