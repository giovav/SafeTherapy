# File: src/ml/predictor.py

"""
Modulo di Inferenza in Real-Time.
Carica in memoria i modelli serializzati e fornisce le predizioni probabilistiche
sul rischio di reazione avversa durante l'esplorazione dell'algoritmo A*.
"""

import os
import pandas as pd
import joblib

class RiskPredictor:
    """
    Gestisce l'istanza del Random Forest addestrato e applica la medesima 
    pipeline di codifica (LabelEncoding) sui dati forniti in fase di query.
    """

    def __init__(self):
        """Costruisce i path di sistema e carica i modelli in memoria."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, "src", "ml", "models")
        
        self.model_path = os.path.join(self.model_dir, "rf_risk_model.pkl")
        self.encoder_path = os.path.join(self.model_dir, "label_encoders.pkl")
        
        self.model = None
        self.encoders = {}
        
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """
        Carica i file `.pkl` generati dal processo di training.
        
        Returns:
            bool: True se il caricamento ha successo, False altrimenti.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.encoder_path):
            print(f"[ML-ERROR] Artifacts non trovati in {self.model_dir}. Eseguire prima il training.")
            return False
            
        self.model = joblib.load(self.model_path)
        self.encoders = joblib.load(self.encoder_path)
        return True

    def predict_risk(self, age: float, sex: str, weight: float, drug_name: str, concomitant: list) -> float:
        """
        Esegue un'inferenza probabilistica sul rischio di effetti avversi per
        pazienti polipatologici. Valuta iterativamente ogni patologia e 
        restituisce il rischio massimo (worst-case scenario clinico).

        Args:
            age (float): Età del paziente.
            sex (str): Sesso biologico.
            weight (float): Peso corporeo in Kg.
            drug_name (str): Molecola da valutare.
            concomitant (list): Lista di patologie o farmaci assunti dal paziente.

        Returns:
            float: Probabilità massima stimata (tra 0.0 e 1.0) di reazione avversa.
        """
        if not self.model:
            return 0.5 

        if isinstance(concomitant, str):
            concomitant = [concomitant]
        if not concomitant:
            concomitant = ['none']

        max_risk = 0.0

        for conc in concomitant:
            input_data = pd.DataFrame([{
                'AGE': age,
                'SEX': sex,
                'WEIGHT': weight,
                'DRUG_NAME': drug_name,
                'CONCOMITANT': conc.strip()
            }])
            
            categorical_cols = ['SEX', 'DRUG_NAME', 'CONCOMITANT']
            
            for col in categorical_cols:
                val = str(input_data[col].iloc[0])
                le = self.encoders.get(col)
                
                if le:
                    # Gestione Out-Of-Vocabulary: fallback alla classe zero
                    if val in le.classes_:
                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0 
                else:
                    input_data[col] = 0
                    
            try:
                risk_prob = self.model.predict_proba(input_data)[0][1]
                if risk_prob > max_risk:
                    max_risk = float(risk_prob)
            except Exception as e:
                print(f"[ML-WARN] Fallimento inferenza sul nodo: {e}")
                
        return max_risk if max_risk > 0.0 else 0.5