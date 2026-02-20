# File: src/sss/heuristic.py

"""
Modulo Euristico per l'Ottimizzazione della Ricerca.
Gestisce l'integrazione tra l'algoritmo di ricerca A* e i modelli
di Intelligenza Artificiale (Machine Learning e Reti Bayesiane).
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ml.predictor import RiskPredictor
from src.bn.learner import FaersBN


class AIHeuristic:
    """
    Motore di valutazione dei costi g(n) e delle stime euristiche h(n).

    Fonde due sorgenti di rischio clinico distinte e complementari:

    - **Rischio molecolare** (Random Forest): stima la probabilità di reazione
      avversa per uno specifico farmaco dato il profilo del paziente.
    - **Fragilità sistemica** (Rete Bayesiana): quantifica la vulnerabilità
      intrinseca del paziente indipendentemente dal farmaco, amplificando
      il costo di tutti i farmaci proposti per pazienti ad alto rischio.

    Il risultato combinato è usato dall'algoritmo A* in `search.py` per
    guidare la ricerca verso terapie più sicure per quel paziente specifico.

    Attributes:
        ml (RiskPredictor): Classificatore Random Forest per il rischio molecolare.
        bn (FaersBN): Rete Bayesiana per la fragilità sistemica del paziente.
        atom_mapping (dict): Dizionario di traduzione atomo Prolog → nome originale.
    """

    def __init__(self):
        """
        Inizializza i modelli predittivi e carica il dizionario di traduzione.

        Carica `RiskPredictor` e `FaersBN` per l'inferenza probabilistica in tempo reale.
        Se il file `atom_mapping.json` è presente, lo legge per consentire
        la traduzione degli atomi Prolog nei nomi farmaceutici attesi dal ML.
        In assenza del file, il mapping rimane vuoto e `_get_original_name`
        restituirà l'atomo grezzo.
        """
        self.ml = RiskPredictor()
        self.bn = FaersBN()

        self.atom_mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mapping_path = os.path.join(base_dir, "kb", "prolog", "atom_mapping.json")

        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.atom_mapping = json.load(f)

    def _get_original_name(self, atom: str) -> str:
        """
        Traduce un atomo Prolog nel nome farmaceutico originale atteso dal modello ML.

        Gli atomi Prolog sono in formato snake_case normalizzato (es. 'acetyl_salicylic_acid'),
        mentre i modelli ML sono stati addestrati sui nomi originali (es. 'acetyl salicylic acid').
        Questo metodo funge da ponte tra i due spazi di rappresentazione.

        Args:
            atom (str): L'atomo Prolog da tradurre.

        Returns:
            str: Il nome originale dalla Knowledge Base se presente nel mapping,
                oppure l'atomo invariato se non viene trovata corrispondenza.
        """
        return self.atom_mapping.get(atom, atom)

    def evaluate_drug_penalty(self, patient_profile: dict, drug_atom: str) -> float:
        """
        Calcola la penalità di rischio clinico reale da sommare al costo g(n).

        Combina la probabilità di reazione avversa specifica per farmaco e paziente
        (modello ML) con l'indice di fragilità sistemica (Rete Bayesiana) tramite
        la formula: `(risk_ml * 1000) * (1 + frailty_bn)`.

        Il moltiplicatore `(1 + frailty_bn)` garantisce che pazienti fragili
        (anziani, sottopeso, polipatologici) abbiano penalità più elevate per
        tutti i farmaci, spingendo A* verso terapie più conservative.

        In caso di errore in uno dei due modelli, il valore di fallback è 0.5
        (rischio neutro), per non bloccare la ricerca.

        Args:
            patient_profile (dict): Profilo clinico del paziente con le chiavi 'age', 
                                    'sex', 'weight' e 'concomitant'.
            drug_atom (str): Atomo Prolog del farmaco da valutare.

        Returns:
            float: Valore di penalità non negativo da aggiungere a g(n).
        """
        drug_real_name = self._get_original_name(drug_atom)

        try:
            risk_ml = self.ml.predict_risk(
                age=patient_profile['age'],
                sex=patient_profile['sex'],
                weight=patient_profile['weight'],
                drug_name=drug_real_name,
                concomitant=patient_profile['concomitant']
            )
        except Exception:
            risk_ml = 0.5

        try:
            frailty_bn = self.bn.get_patient_fragility(
                age=patient_profile['age'],
                weight=patient_profile['weight'],
                concomitant=patient_profile['concomitant']
            )
        except Exception:
            frailty_bn = 0.5

        base_risk_cost = risk_ml * 1000.0
        frailty_multiplier = 1.0 + frailty_bn

        return base_risk_cost * frailty_multiplier

    def calculate_admissible_h(self, remaining_diseases: list) -> float:
        """
        Calcola l'euristica h(n) ammissibile per il problema Set Cover Multi-Target.

        Per garantire la completezza e l'ottimalità dell'algoritmo A*, l'euristica 
        deve essere strettamente ammissibile (non deve mai sovrastimare il costo).
        Assumiamo il rilassamento del caso migliore: esiste un singolo farmaco "miracoloso" 
        a rischio nullo e di prima scelta in grado di curare tutte le patologie rimanenti.
        
        In questo scenario ideale, il costo minimo assoluto per coprire qualsiasi numero 
        di malattie residue coincide con il costo fisso di prescrizione (penalità di 
        polifarmacia) per quel singolo farmaco.

        Args:
            remaining_diseases (list[str]): Lista degli atomi Prolog delle
                patologie ancora da coprire al nodo corrente.

        Returns:
            float: 20.0 (costo base di inserimento farmaco) se ci sono patologie, 
                   0.0 se il goal è stato raggiunto.
        """
        if not remaining_diseases:
            return 0.0

        # Ritorna la penalità fissa minima di polifarmacia impostata in search.py
        return 20.0