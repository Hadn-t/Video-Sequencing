import tensorflow as tf

def convert_to_tflite(model_path, tflite_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set to select TF ops to handle Flex ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Disable experimental lowering of tensor list ops
    converter._experimental_lower_tensor_list_ops = False
    
    # Perform conversion
    tflite_model = converter.convert()
    
    # Save the TFLite model to file
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model saved to {tflite_path}')


