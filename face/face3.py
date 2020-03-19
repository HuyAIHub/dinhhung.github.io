import cv2
import dlib
import sqlite3 as lite
import sys
import os

def face_crop(photo):
    photo1 = 'data/' + photo
    # Load cnn_face_detector with 'mmod_face_detector'
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    # Load image
    img = cv2.imread(photo1)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find faces in image
    rects = dnnFaceDetector(gray, 1)
    left, top, right, bottom = 0, 0, 0, 0

    # For each face 'rect' provides face location in image as pixel loaction
    for (i, rect) in enumerate(rects):
        left = rect.rect.left()  # x1
        top = rect.rect.top()  # y1
        right = rect.rect.right()  # x2
        bottom = rect.rect.bottom()  # y2
        width = right - left
        height = bottom - top

        # Crop image
        img_crop = img[top:top + height, left:left + width]
        name_img = 'img_crop/' + photo

        # save crop image with person name as image name
        cv2.imwrite(name_img, img_crop)

path = os.path.dirname(__file__) + "\\data.db"
con = lite.connect(path)

with con:
    cur = con.cursor()
    cur.execute("SELECT DISTINCT Image FROM img")

    rows = cur.fetchall()

    for row in rows:
        for i in row:
            face_crop(i)



