from pathlib import Path
import joblib


MODEL_PATH = Path("models/features/best_model.joblib")
PREPROCESSOR_PATH = Path("models/features/preprocessor.joblib")


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


from src.config import settings

model = joblib.load(settings.model_path)
preprocessor = joblib.load(settings.preprocessor_path)