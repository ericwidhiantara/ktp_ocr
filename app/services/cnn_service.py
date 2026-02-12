import os
import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_PATH, "data", "cnn", "model.h5")


class CnnService:
    """CNN-based KTP image detection service."""

    _model = None

    @classmethod
    def load_model(cls):
        """Load the CNN model from disk. Call once at startup."""
        if not os.path.exists(MODEL_PATH):
            print(f"[CnnService] WARNING: Model file not found at {MODEL_PATH}")
            print("[CnnService] KTP detection will be skipped.")
            cls._model = None
            return

        from keras.models import load_model as keras_load_model

        cls._model = keras_load_model(MODEL_PATH, compile=False)
        cls._model.make_predict_function()
        print("[CnnService] CNN model loaded successfully.")

    @classmethod
    def is_ktp(cls, pil_image) -> bool:
        """
        Predict whether the given PIL image is a KTP.
        Returns True if KTP is detected, False otherwise.
        If model is not loaded, returns True (skip detection).
        """
        if cls._model is None:
            print("[CnnService] Model not loaded, skipping KTP detection.")
            return True

        img = pil_image.resize((150, 150))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        prediction = cls._model.predict(img)

        print(f"[CnnService] Prediction: {prediction}")

        # 0 means KTP is detected
        return prediction[0][0] == 0
