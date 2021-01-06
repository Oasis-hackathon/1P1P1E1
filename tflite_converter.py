import tensorflow as tf
from tensorflow.keras.models import load_model



model = load_model('model/checkpoint-epoch-100-batch-4-trial-dataset1.h5')
print(model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('test.tflite', 'wb') as f:
    f.write(tflite_model)