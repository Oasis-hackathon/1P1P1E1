import cv2


capture = cv2.VideoCapture(0)  # use 0 for web camera
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)



