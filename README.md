# " _**MASK DETECTION SYSTEM**_ " 프로젝트

--------------------------------------



# 0. **Description** 

![img1](https://github.com/Oasis-hackathon/1P1P1E1/blob/master/img/img1.png)

얼굴인식, 마스크 착용 유무 판단, 글자 인식 등을 이용하여 마스크 미착용자에게 경고 조치를 하는 프로젝트 입니다.

카메라(web cam, camera module)의 이미지를 Raspberry Pi에서 Google Edge TPU를 통해 이용하여 **저사양 임베디드 시스템**에서도 **인공지능 추론(inference)**을 제공하는 웹서비스입니다.

연산량이 가장 높은 추론을 Edge TPU가 대신 하기 때문에, 라즈베리파이의 CPU 부담이 줄어든다.

좌석의 번호를 인식하여 마스크를 착용하지 않은 인원에게 해당 번호로 방송을 합니다.



### **1. Mask Detection System at A.I**

- ### Raspberry Pi4

<img src="https://github.com/Oasis-hackathon/1P1P1E1/blob/master/img/rasp.jpg" alt="rasp" style="zoom: 33%;" />

- ### Google Edge TPU

<img src="https://github.com/Oasis-hackathon/1P1P1E1/blob/master/img/coral.jpg" alt="coral" style="zoom:33%;" />

- ### Camera module, Web Cam

<img src="C:\Users\Mangnani\Desktop\img\camera.jpg" alt="camera" style="zoom:33%;" /><img src="C:\Users\Mangnani\Desktop\img\webcam.jpg" alt="webcam" style="zoom:33%;" />

- ### Speaker

<img src="C:\Users\Mangnani\Desktop\img\speaker.jpg" alt="speaker" style="zoom:33%;" />



### 2. IDE

* Visual Studio Code
* Samba
* SSH



### 3. Language

- python3.7
- Frontend : HTML/CSS/JavaScript/AJAX
- Backend : Flask1.1

### 4. Table of Contents

1. Installation

2. Usage

3. Execute
4. Result





# 1. Installation

> ```
> pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
> pip3 install -r requirements.txt
> ```



# 2. Usage

> 웹을 통해 카메라 Control 및 Monitoring

### main page

<img src="C:\Users\Mangnani\Desktop\img\img1.png" alt="img1" style="zoom: 33%;" />



### monitor 상단 page

<img src="C:\Users\Mangnani\Desktop\img\img2.png" alt="img2" style="zoom:33%;" />



### monitor  page & Controll

![img3](C:\Users\Mangnani\Desktop\img\img3.png)

### Mask Detect A.I  page

<img src="C:\Users\Mangnani\Desktop\img\img4.png" alt="img4" style="zoom:33%;" />

# 3. Execute

> python3 main_server.py

![algo](C:\Users\Mangnani\Desktop\img\algo.png)



# 4. Result

> Keras Convoluton Network Train & Validation Result

<img src="C:\Users\Mangnani\Desktop\img\mobileNetV2.png" alt="mobileNetV2" style="zoom: 33%;" /><img src="C:\Users\Mangnani\Desktop\img\ResNet.png" alt="ResNet" style="zoom: 33%;" /><img src="C:\Users\Mangnani\Desktop\img\DenseNet.png" alt="DenseNet" style="zoom: 33%;" />



> Tflite 
>
> 컴퓨터 비전 보다 더 멀리 어두운 곳에서도 높은 정확률을 보임

## distance : 1m

![result1](C:\Users\Mangnani\Desktop\img\result1.png)

## distance : 5m

![result](C:\Users\Mangnani\Desktop\img\result.png)





> **Link**

https://coral.ai/docs/edgetpu/models-intro/



**Creater**

* SungJun.Kwon
* JungSun.Yun
* WooSung.Kong
