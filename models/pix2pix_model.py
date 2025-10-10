import tensorflow as tf
import numpy as np
import cv2

class Pix2PixModel:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.infer = self.model.signatures["serving_default"]  # ✅ Get serving function

    def preprocess(self, img):
        # Convert BGR (OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 1: Resize to 286×286 (matching training jitter step)
        img = cv2.resize(img, (286, 286), interpolation=cv2.INTER_NEAREST)

        # Step 2: Center crop to 256×256
        start = (286 - 256) // 2
        img = img[start:start+256, start:start+256]

        # Step 3: Normalize to [-1, 1]
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0

        # Step 4: Add batch dimension
        img = np.expand_dims(img, axis=0)
        return tf.convert_to_tensor(img)

    def postprocess(self, img_tensor):
        # Convert back to [0, 255] range and uint8
        img = img_tensor[0].numpy()
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Convert RGB back to BGR for OpenCV display/save
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def predict(self, img):
        # Preprocess the input image
        input_tensor = self.preprocess(img)

        # Run inference through the model
        output_dict = self.infer(input_tensor)
        output_tensor = list(output_dict.values())[0]  # ✅ Get actual output tensor

        # Postprocess and return the result
        return self.postprocess(output_tensor)
