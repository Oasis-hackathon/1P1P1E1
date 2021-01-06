from flask import Blueprint, render_template, Response, url_for
import detect
import norm
import inference
from PIL import Image, ImageFont
from PIL import ImageDraw
import cv2
import cv2_camera
import numpy as np
import subprocess
import pytesseract


# blueprint
tf_face_tracking = Blueprint('tf', __name__)

interpreter = inference.interpreter
interpreter_mask = inference.interpreter_mask

cmd = 'gtts-cli '
contents = '\"마스크를 써주시길 바랍니다.\" '
option = '--lang ko | mpg123 -'
word_list = ["1 번", "2 번", "3 번"]
            
# Draws the bounding boxes function
def draw_objects(image, objs):
    for obj in objs:
        bbox = obj.bbox
        
        inf_img = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        
        if inf_img.size:
            #img = cv2.cvtColor(inf_img, cv2.COLOR_BGR2RGB)
            #img = cv2.filter2D(image, -1, norm.sharpening_2)
            img = cv2.dnn.blobFromImage(inf_img, scalefactor=1., size=(224, 224), mean=(104., 177., 123.))
            img = np.transpose(img.squeeze(), (1,2,0))
            
            # preprocessing( -1 ~ 1 )
            preprocess_img = norm.stack(norm.scale(img, -1, 1))            
            preprocess_img = np.expand_dims(preprocess_img, axis=0)
            
            # inferecne
            inference.interpreter_invoke(interpreter_mask, preprocess_img)

            output_data = interpreter_mask.tensor(interpreter_mask.get_output_details()[0]['index'])
            mask, nomask = output_data().squeeze()
            

            if mask >= nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                """
                subprocess.call(cmd+contents+option, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                result = pytesseract.image_to_string(image, config=custom_config, lang='Hangul')
                   if result:
                    for word in word_list:
                        if result in word:
                            print(result)
                    else:
                        value = None
                """
                color = (255, 0, 0)
                label = 'No Mask %d%%' % (nomask * 100)
                cv2.putText(image, text="No mask", org=(0, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=color, thickness=1, lineType=cv2.LINE_AA)
                
                    

            cv2.rectangle(image,(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color,1)
            cv2.putText(image, text=label, org=(bbox.xmin, bbox.ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
                        


# inference
def gen_frames():
    
    while True:
        ret, image = cv2_camera.capture.read()
        

        # image = cv2.flip(image, 1) # 상하반전
        # img = cv2.transpose(image) # 좌우반전
        #image = cv2.flip(image, 1)
        # image reshape
        image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        inference.interpreter_invoke(interpreter, image)

        objs = detect.get_output(interpreter, 0.02, (1.0, 1.0))

        if len(image):
            draw_objects(image, objs)

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@tf_face_tracking.route('/video_feed2')
def tf_tracking():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@tf_face_tracking.route('/cam2', methods=['POST'])
def index():
    return render_template('home2.html')


@tf_face_tracking.route('/button1', methods=['POST'])
def button_func():
    return render_template('home.html')


