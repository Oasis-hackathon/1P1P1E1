import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from imutils import paths


# 사용가능 한 GPU 확인
print(device_lib.list_local_devices())

# learning rate, Epoch, Batch size  초기화
INIT_LR = 1e-4
EPOCHS = 5
BS = 8

# PATH

# PATH
model_dir = 'model'
dataset_dir = 'dataset2'
plot_dir = 'plot'

base_path = os.getcwd()
model_path = os.path.join(base_path, model_dir)
dataset_path = os.path.join(base_path, dataset_dir)
plot_path = os.path.join(base_path, plot_dir)

filename = 'checkpoint-epoch-{}-batch-{}-transfer_learning-{}.h5'.format(EPOCHS, BS, dataset_dir)
plot_name = 'checkpoint-epoch-{}-batch-{}-transfer_learning-{}.png'.format(EPOCHS, BS, dataset_dir)

model_save_path = os.path.join(model_path, filename)
plot_save_path = os.path.join(plot_path, plot_name)

# checkpoint
checkpoint = ModelCheckpoint(model_save_path,             # file명을 지정
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출
                             verbose=1,            # 로그를 출력
                             save_best_only=True,  # 가장 best 값만 저장
                             mode='auto'           # auto는 알아서 best를 찾음
                            )

# dataset을 directory에서 이미지 목록을 가져옴
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []

# 이미지 경로 반복 처리
for imagePath in imagePaths:
	# 파일 이름에서 클래스 레이블 추출
	label = imagePath.split(os.path.sep)[-2]

	# 입력 이미지 (224x224) 로드 및 전처라
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# 데이터 및 레이블 목록을 각각 업데이트
	data.append(image)
	labels.append(label)

# 데이터와 레이블을 Numpy 배열로 변환
data = np.array(data, dtype="float32")
labels = np.array(labels)

#  one-hot encoding 수행
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# 80% 훈련용 20% 테스트용
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 훈련 이미지 생성
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


# load model
model = load_model('model/checkpoint-epoch-100-batch-4-trial-dataset1.h5')
model.summary()


with tf.device('/gpu:0'):
    model = Model(inputs=model.input, outputs=model.output)

    for layer in model.layers:
        layer.trainable = False

    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # Train model
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

# model predict
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 가장 확률이 높은 index 찾기
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# print("[INFO] saving mask detector model...")
# model.save('model.h5', save_format="h5")

print("[INFO] plot the training loss and accuracy")
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig(plot_save_path)
