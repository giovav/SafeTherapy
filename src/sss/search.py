# File: src/sss/search.py
import heapq
import os
import sys
import json
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.kb.interface import PrologInterface
from src.kb.utils import to_prolog_atom
from src.sss.heuristic import AIHeuristic

class TherapyNode:
    """
    Rappresenta uno stato (nodo) all'interno dell'albero di ricerca A*.
    Contiene la terapia parziale e il conteggio dei costi per guidare l'esplorazione.
    
    Attributes:
        selected_drugs (dict): Mappa {farmaco_atom: set(malattie_coperte)}.
        remaining_diseases (frozenset): Insieme delle patologie non ancora trattate.
        g (float): Costo reale accumulato (penalità cliniche, polifarmacia, rischio ML/BN).
        h (float): Stima euristica ammissibile del costo rimanente verso il goal.
        f (float): Costo totale stimato del nodo (f = g + h).
    """
    def __init__(self, selected_drugs: dict, remaining_diseases: frozenset, g: float, h: float):
        """Inizializza un nodo dell'albero di ricerca con i relativi costi."""
        self.selected_drugs = selected_drugs 
        self.remaining_diseases = remaining_diseases 
        self.g = g 
        self.h = h 
        self.f = g + h 

    def __lt__(self, other) -> bool:
        """
        Metodo di comparazione per la gestione della Priority Queue (Min-Heap).
        L'A* estrae sempre il nodo con il costo f(n) minore.
        """
        return self.f < other.f

class TherapyOptimizer:
    """
    Motore di ottimizzazione principale basato sull'algoritmo A*.
    Interroga la T-Box (Prolog) per le regole cliniche assolute e valuta 
    i percorsi probabilistici tramite l'euristica Neuro-Simbolica (ML + BBN).
    """
    def __init__(self):
        """
        Inizializza le interfacce verso la Knowledge Base Prolog e i modelli AI.
        Carica inoltre il mapping degli atomi per la traduzione dei nomi.
        """
        print("[SSS] Inizializzazione Algoritmo A* (Ontological Set Cover Mode)...")
        self.kb = PrologInterface()
        self.ai = AIHeuristic()
        self.polypharmacy_penalty = 20.0
        self.atom_mapping = {}
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mapping_path = os.path.join(base_dir, "kb", "prolog", "atom_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.atom_mapping = json.load(f)

    def _get_candidates_for_disease(self, disease_atom: str) -> set:
        """
        Estrae dalla T-Box i farmaci le cui proprietà biologiche (farmacodinamica)
        soddisfano i requisiti fisiopatologici della malattia.
        
        Args:
            disease_atom (str): L'atomo Prolog della patologia da curare.
            
        Returns:
            set: Insieme degli atomi Prolog dei farmaci candidati.
        """
        candidates = set()
        query_result = list(self.kb.prolog.query(f"approved_for(Drug, '{disease_atom}', _)"))
        for res in query_result:
            candidates.add(res['Drug'])
        return candidates

    def _get_covered_diseases(self, drug_atom: str, target_diseases: frozenset) -> set:
        """
        Verifica quali patologie target vengono trattate dal farmaco fornito
        sfruttando le capacità di ragionamento ontologico (pleiotropia).
        
        Args:
            drug_atom (str): L'atomo Prolog del farmaco.
            target_diseases (frozenset): Le patologie da testare.
            
        Returns:
            set: Sottoinsieme delle patologie coperte dal farmaco.
        """
        covered = set()
        for disease in target_diseases:
            if list(self.kb.prolog.query(f"approved_for('{drug_atom}', '{disease}', _)")):
                covered.add(disease)
        return covered

    def _get_disease_specific_cost(self, drug_atom: str, target_disease: str) -> float:
        """
        Calcola il costo prescrittivo dando priorità alla linea guida clinica.
        Interroga Prolog per la Linea di trattamento (1=Prima scelta, 2=Seconda, 3=Estrema ratio).
        
        Args:
            drug_atom (str): Il farmaco da valutare.
            target_disease (str): La patologia bersaglio.
            
        Returns:
            float: Costo algoritmico basato sull'appropriatezza clinica.
        """
        try:
            res = list(self.kb.prolog.query(f"approved_for('{drug_atom}', '{target_disease}', Line)"))
            if res:
                line = int(res[0]['Line'])
                
                if line == 1:
                    return 0.0
                elif line == 2:
                    return 2000.0
                elif line == 3:
                    return 4000.0
        except Exception:
            pass
        
        return 10000.0

    def _calculate_safety_penalty(self, current_drugs: dict, new_drug_atom: str) -> float:
        """
        Interroga la T-Box (Prolog/FOL) per rilevare interazioni farmacologiche (DDI)
        e calcola la conseguente penalità sul costo reale g(n).
        
        Args:
            current_drugs (dict): I farmaci già prescritti nello stato corrente.
            new_drug_atom (str): Il nuovo farmaco da aggiungere alla combinazione.
            
        Returns:
            float: Penalità per l'A*. Restituisce float('inf') in caso di 
                   controindicazioni assolute (pruning del ramo).
        """
        drugs_to_test = list(current_drugs.keys()) + [new_drug_atom]
        if len(drugs_to_test) < 2: 
            return 0.0
            
        validation = self.kb.verify_therapy(drugs_to_test)
        if not validation['safe']:
            penalty = 0.0
            for conflict in validation.get('conflicts', []):
                severity = conflict.get('severity', 'unknown')
                if severity == 'high': 
                    return float('inf')
                elif severity == 'medium': 
                    penalty += 500.0
            return penalty
        return 0.0

    def solve(self, patient_profile: dict, target_diseases: list) -> TherapyNode:
        """
        Esegue l'algoritmo A* esplorando lo spazio logico della T-Box per trovare
        la combinazione farmacologica ottima (minimo rischio globale) che copre
        tutte le patologie target.

        Args:
            patient_profile (dict): Profilo clinico del paziente (usato dai modelli ML/BBN).
            target_diseases (list): Lista delle patologie testuali da curare.

        Returns:
            TherapyNode: Il nodo terminale contenente la terapia ottima e il suo costo, 
                         oppure None se non esiste alcuna soluzione sicura.
        """
        valid_disease_atoms = set()
        for d in target_diseases:
            atom = to_prolog_atom(d)
            if not self._get_candidates_for_disease(atom):
                print(f"[SSS-WARN] Patologia '{d}' non riconosciuta o priva di cure nella T-Box. Ignorata.")
            else:
                valid_disease_atoms.add(atom)
                
        disease_atoms = frozenset(valid_disease_atoms)
        if not disease_atoms: 
            print("[SSS-ERROR] Nessuna patologia curabile fornita.")
            return None
        
        open_list = []
        visited_states = {} 
        
        start_node = TherapyNode(selected_drugs={}, remaining_diseases=disease_atoms, g=0.0, h=0.0)
        heapq.heappush(open_list, start_node)
        
        print("[SSS] Avvio ricerca A* nello spazio ontologico (T-Box)...")
        
        while open_list:
            current_node = heapq.heappop(open_list)
            
            if not current_node.remaining_diseases:
                return current_node
                
            target = next(iter(current_node.remaining_diseases))
            candidates = self._get_candidates_for_disease(target)
            
            if not candidates: 
                continue
                
            for drug in candidates:
                new_selected = copy.deepcopy(current_node.selected_drugs)
                covered_diseases = self._get_covered_diseases(drug, current_node.remaining_diseases)
                
                new_remaining = frozenset(current_node.remaining_diseases - covered_diseases)
                
                step_g = 0.0
                if drug not in new_selected:
                    step_g += self.polypharmacy_penalty
                    step_g += self._get_disease_specific_cost(drug, target)
                    step_g += self.ai.evaluate_drug_penalty(patient_profile, drug)
                    
                    safety_penalty = self._calculate_safety_penalty(current_node.selected_drugs, drug)
                    if safety_penalty == float('inf'): 
                        continue
                    step_g += safety_penalty
                
                new_g = current_node.g + step_g
                
                if drug not in new_selected: 
                    new_selected[drug] = set()
                new_selected[drug].update(covered_diseases)
                
                new_h = self.ai.calculate_admissible_h(list(new_remaining))
                new_node = TherapyNode(new_selected, new_remaining, new_g, new_h)
                
                state_sig = (frozenset(new_selected.keys()), new_remaining)
                
                if state_sig not in visited_states or new_g < visited_states[state_sig]:
                    visited_states[state_sig] = new_g
                    heapq.heappush(open_list, new_node)
                    
        return None