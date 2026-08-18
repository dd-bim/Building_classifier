import os
import joblib
from sklearn.preprocessing import LabelEncoder


class LabelEncoderManager:
    """Verwaltet das Laden, Speichern und Bereitstellen der LabelEncoder für kategorische Features."""

    CATEGORICAL_FEATURES = ['roof_type', 'development_type_code', 'neighbour_majority_class', 'building_age']

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.label_encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        self.label_encoders = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.label_encoders_path):
            with open(self.label_encoders_path, 'rb') as f:
                encoders = joblib.load(f)
        else:
            encoders = {}
        for feature in self.CATEGORICAL_FEATURES:
            if feature not in encoders:
                encoders[feature] = LabelEncoder()
        return encoders

    def save_label_encoders(self):
        with open(self.label_encoders_path, 'wb') as f:
            joblib.dump(self.label_encoders, f)

    def get_label_encoders(self) -> dict:
        return self.label_encoders
