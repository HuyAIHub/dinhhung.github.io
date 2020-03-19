import time
import imutils
import dlib
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras import applications
import tensorflow as tf
from keras import backend as K
import h5py

# Đọc ảnh đầu vào
image = cv2.imread('yen.jpg')

# Khai báo việc sử dụng các hàm của dlib
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
'''
# Thực hiện xác định bằng HOG và SVM
start = time.time()
faces_hog = hog_face_detector(image, 1)
end = time.time()
print("Hog + SVM Execution time: " + str(end - start))

# Vẽ một đường bao màu xanh lá xung quanh các khuôn mặt được xác định ra bởi HOG + SVM

for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
'''

# Thực hiện xác định bằng CNN
start = time.time()
faces_cnn = cnn_face_detector(image, 1)
end = time.time()
print("CNN Execution time: " + str(end - start))

# Vẽ một đường bao đỏ xung quanh các khuôn mặt được xác định bởi CNN
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)

def convnet_model_():
    vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=(221, 221, 3))
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x,axis=1))(x)
#     x = Lambda(K.l2_normalize)(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
    convnet_model = convnet_model_()

    first_input = Input(shape=(221, 221, 3))
    first_conv = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(221, 221, 3))
    second_conv = Conv2D(96, kernel_size=(8,8), strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7), strides=(4,4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(128)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model
deep_rank_model = deep_rank_model()
batch_size = 24

_EPSILON = K.epsilon()
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    loss = 0.
    g = 1.
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i]
            p_embedding = y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p = K.sqrt(K.sum((q_embedding - p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
            loss = loss + g + D_q_p - D_q_n
        except:
            continue
    loss = loss/batch_size*3
    return K.maximum(loss, 0)

deep_rank_model.compile(loss=_loss_tensor, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))

def image_batch_generator(images, labels, batch_size):
    labels = np.array(labels)
    while True:
        batch_paths = np.random.choice(a = len(images), size = batch_size//3)
        input_1 = []

        for i in batch_paths:
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]

            j = np.random.choice(pos)
            while j == i:
                j = np.random.choice(pos)

            k = np.random.choice(neg)
            while k == i:
                k = np.random.choice(neg)

            input_1.append(images[i])
            input_1.append(images[j])
            input_1.append(images[k])

        input_1 = np.array(input_1)
        input = [input_1, input_1, input_1]
        yield(input, np.zeros((batch_size, )))
deep_rank_model.fit_generator(generator=image_batch_generator(X, y, batch_size),
                   steps_per_epoch=len(X)//batch_size,
                   epochs=2000,
                   verbose=1,
                   callbacks=callbacks_list)


