import cv2
from keras.models import load_model
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image

model = load_model('kannada_number.h5')
geo = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '-1': '',
    '':''
}

kannada_ref_png = cv2.imread('kannada_numbers.png')
def main():
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([94, 80, 2])
    Upper_green = np.array([126, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = -1

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        #overlap
        img[0:kannada_ref_png.shape[0], 0: kannada_ref_png.shape[1]] = kannada_ref_png
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 2000:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (255, 255, 255), 7)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if blackboard_cnts is not None and len(blackboard_cnts) >= 1:
                    # print('blackboard_cnts', blackboard_cnts)
                    # print('v2.contourArea', cv2.contourArea)
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        pred_probab, pred_class = keras_predict(model, digit)
                        # print(pred_class, pred_probab)

            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        fontpath = "kannada.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((190, 430), str(geo[str(pred_class)]), font=font, fill=(255, 255, 0, 1))
        img = np.array(img_pil)
        cv2.putText(img, "Predicted Number : " + str(geo[str(pred_class)]), (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, "Number in Kannada: ", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()