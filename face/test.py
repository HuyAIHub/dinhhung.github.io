import dlib
import cv2

dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load image
img = cv2.imread('hung.jpg')

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

    # save crop image with person name as image name
    #name = 'Images_crop/' + person + '/' + image_name
    cv2.imwrite('hoai22.jpg', img_crop)
