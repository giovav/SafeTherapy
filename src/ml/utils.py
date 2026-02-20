# File: src/ml/utils.py
import os

class ProjectConfig:
    def __init__(self):
        # Calcola la root del progetto (3 livelli sopra questo file)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Percorsi Dati
        self.data_dir = os.path.join(self.base_dir, "data")
        self.data_path = os.path.join(self.data_dir, "faers_50k_examples.csv") # <--- Il file nuovo
        
        # Percorsi Modelli
        self.model_dir = os.path.join(self.base_dir, "src", "ml", "models")
        self.model_path = os.path.join(self.model_dir, "rf_faers_model.pkl")

    def check_paths(self):
        if not os.path.exists(self.data_path):
            return False, f"Dataset non trovato in: {self.data_path}"
        return True, "OK"

class TextUtils:
    @staticmethod
    def clean_text(text):
        """Pulisce le stringhe per garantire match coerenti (Uppercase)."""
        if not isinstance(text, str): return "NONE"
        return text.strip().upper()