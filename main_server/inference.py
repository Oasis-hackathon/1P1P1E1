import tflite_runtime.interpreter as tflite
import os
import detect


# .tflite interpreter
interpreter = tflite.Interpreter(
    os.path.join(os.getcwd(),"tflite", "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"),
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', {"device": "usb:0"})]
    )   
interpreter.allocate_tensors()

interpreter_mask = tflite.Interpreter(
    os.path.join(os.getcwd(), "tflite", "mask_detectorSH_V3.tflite"),
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', {"device": "usb:0"})]
    )
interpreter_mask.allocate_tensors()

# inferecne
def interpreter_invoke(interpreter, image):
    tensor_mask = detect.input_tensor(interpreter=interpreter)[:, :] = image.copy()
    tensor_mask.fill(0)  # padding        
    interpreter.invoke()
    

