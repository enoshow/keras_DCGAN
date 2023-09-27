import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob
import os


def walk():
    folder = "in_images"
    fol = []
    for cd, dir, files in os.walk(folder):
        # print(cd)
        # print(f"dir : {dir}")
        if dir != []:
            fol = dir
    print(f"fol : {fol}")
    return fol

def high_pass_filter(image_path, save_path):
    # 画像を読み込む
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像をfloat型に変換して正規化
    img_float32 = np.float32(img) / 255.0

    # DFT（離散フーリエ変換）を適用
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 画像のサイズを取得
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # マスクの作成（中心部分を0、それ以外を1とする）
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # 中心部分の半径
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    # High-passフィルタを適用
    fshift = dft_shift * mask
    # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    # 逆DFTを適用して画像を復元
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 画像の保存
    if save_path:
        cv2.imwrite(save_path, img_back * 255)

    return img_back



def high_pass_IMG_filter(img_path):
    # 画像をロード
    original = Image.open(img_path).convert('L')  # 'L' はグレースケールに変換するため

    # Low-passフィルタを適用 (ガウシアンブラーを使用)
    blurred = original.filter(ImageFilter.GaussianBlur(radius=3))

    # High-passフィルタの結果を得るため、ブラードした画像をオリジナル画像から減算
    high_pass = Image.blend(original, blurred, alpha=-1)

    # 結果を[-255, 255]の範囲にクリップして、[0, 255]の範囲にスケーリング
    high_pass = Image.fromarray(np.clip(np.array(high_pass), 0, 255).astype(np.uint8))

    return high_pass

fol = walk()
for f in fol:
    n = 0
    files = glob.glob(fr"./in_images/{f}/*.pgm", recursive=True) 

    if os.path.isdir(f"./in_images/{f}"):
        print(f"'./in_images/{f}' : exist")
    else:
        os.mkdir(f"./in_images/{f}")

    for file in files:
        # 画像のパスを指定して関数を実行
        result_image = high_pass_IMG_filter(file)
        result_image.save(f'./in_images/{f}/in_image_prepro_{n}.pgm')
        n = n + 1


