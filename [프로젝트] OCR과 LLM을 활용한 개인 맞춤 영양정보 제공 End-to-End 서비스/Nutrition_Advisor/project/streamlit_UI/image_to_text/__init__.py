import easyocr
import cv2
import numpy as np

def easy_ocr(preprocessed_img):
    reader = easyocr.Reader(['ko', 'en'], model_storage_directory='korean_g2.pth')
    result = reader.readtext(preprocessed_img)
    text_result = [res[1] for res in result]
    return text_result
    

def image_preprocessing(img):
    img = cv2.resize(img, dsize= None, fx = 2, fy = 2, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 커널 생성(대상이 있는 픽셀을 강조)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    image_sharp = cv2.filter2D(img_bin, -1, kernel)
    # plt.imshow(image_sharp, cmap = 'gray')
    return image_sharp