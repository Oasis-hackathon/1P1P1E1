from flask import Flask, render_template, Response, request, redirect, jsonify
import tf_blueprint
import cv2
import subprocess
import uvcdynctrl
import cv2_camera



app = Flask(__name__)
app.register_blueprint(tf_blueprint.tf_face_tracking)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = cv2_camera.capture.read()  # read the camera frame
        #frame = cv2.transpose(frame)
        #frame = cv2.flip(frame, 1)
        if (not success):
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/main')
def index():    
    return render_template('index.html')

@app.route('/ajax', methods=['POST'])
def ajax():
    data =request.get_json()
    index = int(data['id'])
    
    subprocess.call(uvcdynctrl.cmd[index], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
    return jsonify(result="success", result2=data)


@app.route('/home', methods=['POST'])
def button_func():
    return render_template('home.html')


if (__name__ == '__main__'):
    app.run(host='192.168.100.107', port=8080)#, debug=True)#, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))

