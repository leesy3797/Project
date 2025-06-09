import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from matplotlib import font_manager

def putText(cv_img, text, x, y, color=(0, 0, 0), font_size=22):
  # Colab이 아닌 Local에서 수행 시에는 gulim.ttc 를 사용하면 됩니다.
  # font = ImageFont.truetype("fonts/gulim.ttc", font_size)
  font = ImageFont.truetype('C:\\Windows\\Fonts\\malgun.ttf', 20)
  img = Image.fromarray(cv_img)
  draw = ImageDraw.Draw(img)
  draw.text((x, y), text, font=font, fill=color)
  cv_img = np.array(img)
  return cv_img

reader = easyocr.Reader(['ko', 'en'], model_storage_directory='korean_g2')
font = ImageFont.truetype('C:\\Windows\\Fonts\\malgun.ttf', 20)

cap = cv2.VideoCapture(0)

# print('Frame width:', round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
# print('Frame height:', round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    img = Image.fromarray(frame) 
    draw = ImageDraw.Draw(img)

    result = reader.readtext(frame)
    if result:
        for (bbox, text, prob) in result: 
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            draw.text((tl[0], tl[1] - 10), text = text, font = font, fill = (0,255,0))
            frame = putText(frame, text, tl[0], tl[1] - 60, (0, 255, 0), 50)
            # plt.rcParams['figure.figsize'] = (16,16)
    cv2.imshow('frame', frame)
    # draw.show()
    # inversed = ~frame

    # cv2.imshow('frame', frame)
    # cv2.imshow('inversed', inversed)
    
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()