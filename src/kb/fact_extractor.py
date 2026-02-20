# File: src/kb/fact_extractor.py

"""
Modulo ETL per la generazione della Knowledge Base Estensionale (A-Box).
Elabora il catalogo WHO ATC-DDD estraendo i mapping tra principi attivi 
e classi farmacologiche.
"""

import os
import sys
import json
import pandas as pd

try:
    from .utils import to_prolog_atom
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import to_prolog_atom

class FactsExtractor:
    """
    Gestisce l'estrazione dei fatti Prolog a partire dai dataset governativi
    e crea i dizionari di traduzione necessari ai moduli ML.
    """

    def __init__(self, data_dir: str, output_dir: str):
        """
        Inizializza l'estrattore definendo i path di I/O.

        Args:
            data_dir (str): Directory contenente i dataset raw (.csv).
            output_dir (str): Directory di destinazione per i file Prolog e JSON.
        """
        self.who_csv_path = os.path.join(data_dir, "WHO ATC-DDD 2024-07-31.csv")
        self.output_file = os.path.join(output_dir, "facts.pl")
        self.mapping_file = os.path.join(output_dir, "atom_mapping.json")
        
        self.facts = set()
        self.atom_mapping = {}

    def process_who_catalog(self) -> None:
        """
        Elabora il dataset WHO ATC, genera i predicati has_atc_code/2
        e compila il dizionario di traduzione atomo-stringa.
        """
        if not os.path.exists(self.who_csv_path):
            print(f"[ERROR] File WHO non trovato: {self.who_csv_path}")
            sys.exit(1)

        df = pd.read_csv(self.who_csv_path)
        atc_col = 'atc_code' if 'atc_code' in df.columns else df.columns[0]
        name_col = 'atc_name' if 'atc_name' in df.columns else df.columns[1]

        df_valid = df.dropna(subset=[atc_col, name_col])
        df_valid = df_valid[df_valid[atc_col].str.len() == 7]

        for _, row in df_valid.iterrows():
            original_name = str(row[name_col]).strip().lower()
            atc_code = str(row[atc_col]).strip().lower()
            drug_atom = to_prolog_atom(original_name)
            
            if drug_atom != "unknown" and atc_code:
                self.facts.add(f"has_atc_code('{drug_atom}', '{atc_code}').")
                self.atom_mapping[drug_atom] = original_name

    def save_artifacts(self) -> None:
        """
        Salva fisicamente su disco la Knowledge Base in formato Prolog
        e il file di mapping in formato JSON.
        """
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("% --- EXTENSIONAL KNOWLEDGE BASE (A-BOX) ---\n")
            f.write(":- multifile has_atc_code/2.\n")
            f.write(":- discontiguous has_atc_code/2.\n\n")
            for fact in sorted(self.facts):
                f.write(fact + "\n")

        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.atom_mapping, f, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "src", "kb", "prolog")
    
    extractor = FactsExtractor(data_dir, output_dir)
    extractor.process_who_catalog()
    extractor.save_artifacts()