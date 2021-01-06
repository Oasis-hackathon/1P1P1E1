import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121 # densenet

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input  # preprocessing image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


def preproccess_image(image_path, img_height, img_width):
	img = load_img(image_path, target_size=(img_height, img_width))
	img = img_to_array(img)
	img = preprocess_input(img)
	# img = np.expand_dims(img, axis=0)
	return img


# 사용가능 한 GPU 확인
print(device_lib.list_local_devices())

# learning rate, Epoch, Batch size  초기화
INIT_LR = 1e-4
EPOCHS = 10
BS = 4

# PATH
model_dir = 'model'
dataset_dir = 'dataset1'
plot_dir = 'plot'

base_path = os.getcwd()
model_path = os.path.join(base_path, model_dir)
dataset_path = os.path.join(base_path, dataset_dir)
plot_path = os.path.join(base_path, plot_dir)

filename = 'checkpoint-epoch-{}-batch-{}-trial-{}.h5'.format(EPOCHS, BS, dataset_dir)
plot_name = 'checkpoint-epoch-{}-batch-{}-trial-{}.png'.format(EPOCHS, BS, dataset_dir)

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

	# 입력 이미지 (224x224) 로드 및 전처리
	image = preproccess_image(imagePath, 224, 224)

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

with tf.device('/gpu:0'):
	# MobileNetV2 network
	baseModel = DenseNet121(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	model = Model(inputs=baseModel.input, outputs=headModel)

	for layer in baseModel.layers:
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
	print("[INFO] val_loss :,", H.history["val_loss"][0], "val_acc", H.history["val_accuracy"][0])

# model predict
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 가장 확률이 높은 index 찾기
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# print("[INFO] saving mask detector model...")
# model.save('model.h5', save_format="h5")

print("[INFO] plot the training loss and accuracy")
# print(H.history)
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("DenseNet121 Loss and Accuracy")
plt.xlabel("Epoch 10")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig(plot_save_path)
