import dlib
import cv2
import numpy as np
import sys

if __name__ == "__main__":

    param = sys.argv
    train_param = param[2]

    detector = dlib.simple_object_detector(train_param)
    # 画像の読み込み
    img = cv2.imread(param[1])

    # 検出箇所を矩形で描画
    rectangles = detector(img)
    for rect in rectangles:
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (147, 20, 255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
