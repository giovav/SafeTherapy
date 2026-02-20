# File: src/bn/predictor.py

import os
import joblib
import pandas as pd
from pgmpy.inference import VariableElimination

class BNPredictor:
    """
    Modulo di Inferenza in Real-Time per la Rete Bayesiana.
    
    Carica in memoria il modello della Rete Bayesiana precedentemente addestrato
    e serializzato dal learner. Fornisce le stime probabilistiche sulla fragilità
    del paziente durante l'esplorazione dell'algoritmo A*.
    """

    def __init__(self):
        """
        Inizializza i percorsi di sistema e carica il modello Bayesiano serializzato.
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(self.base_dir, "src", "bn", "models", "faers_frailty_bbn.pkl")
        
        self.network = None
        self.inference_engine = None
        
        self._load_model()

    def _load_model(self) -> None:
        """
        Tenta di caricare il modello `.pkl` salvato dal learner.
        """
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.network = data.get('network')
                if self.network:
                    self.inference_engine = VariableElimination(self.network)
            except Exception as e:
                print(f"[BN-PREDICT] Errore nel caricamento del modello Bayesiano: {e}")
        else:
            print(f"[BN-PREDICT] Modello non trovato in {self.model_path}. Eseguire prima il learner.")

    def _discretize_age(self, age: float) -> str:
        """Discretizza l'età continua nella corrispondente categoria clinica."""
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        else:
            return 'geriatric'

    def _discretize_weight(self, weight: float) -> str:
        """Discretizza il peso corporeo nella corrispondente categoria metabolica."""
        if weight < 50:
            return 'underweight'
        elif weight <= 90:
            return 'normal'
        else:
            return 'overweight'

    def get_patient_fragility(self, age: float, weight: float, concomitant: list) -> float:
        """
        Calcola l'indice di fragilità sistemica del paziente tramite inferenza.

        Args:
            age (float): Età del paziente in anni.
            weight (float): Peso corporeo in chilogrammi.
            concomitant (list): Anamnesi del paziente.

        Returns:
            float: Probabilità P(IsFragile=1 | evidenza) in range [0.0, 1.0].
        """
        if not self.inference_engine:
            return 0.5

        if isinstance(concomitant, str):
            concomitant = [concomitant]

        age_group = self._discretize_age(age)
        weight_group = self._discretize_weight(weight)
        
        has_conc = "0"
        for c in concomitant:
            if c and c.lower().strip() != 'none':
                has_conc = "1"
                break

        try:
            query_result = self.inference_engine.query(
                variables=['IsFragile'],
                evidence={
                    'AgeGroup': age_group,
                    'WeightGroup': weight_group,
                    'HasConcomitant': has_conc
                },
                show_progress=False
            )
            state_idx = query_result.state_names['IsFragile'].index("1")
            return float(query_result.values[state_idx])
        except Exception as e:
            print(f"[BN-PREDICT] Errore inferenza Bayesiana: {e}. Fallback a 0.5")
            return 0.5