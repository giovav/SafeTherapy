# File: src/main.py

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.sss.search import TherapyOptimizer

def format_disease_names(optimizer: TherapyOptimizer, disease_atoms: set) -> str:
    """
    Traduce un insieme di atomi Prolog (es. 'coronary_artery_disease') nei 
    rispettivi nomi originali formattati (es. 'Coronary artery disease').
    
    Args:
        optimizer (TherapyOptimizer): L'istanza dell'ottimizzatore contenente il mapping.
        disease_atoms (set): L'insieme degli atomi delle patologie coperte dal farmaco.
        
    Returns:
        str: Una stringa leggibile con le patologie separate da virgola.
    """
    real_names = [optimizer.ai._get_original_name(d) for d in disease_atoms]
    return ", ".join(sorted(real_names))

def main():
    """
    Entry point principale dell'agente clinico (CLI).
    Raccoglie i dati del paziente, costruisce la query, avvia la ricerca 
    nello spazio degli stati (A* Set Cover) e formatta l'output terapeutico.
    """
    parser = argparse.ArgumentParser(description="SafeTherapy AI - CDSS Optimization Engine")
    
    parser.add_argument("--age", type=int, required=True, help="Età del paziente (es. 65)")
    parser.add_argument("--weight", type=float, required=True, help="Peso del paziente in kg (es. 80.5)")
    parser.add_argument("--sex", type=str, choices=['M', 'F'], default='M', help="Sesso del paziente (M/F)")
    parser.add_argument("--conditions", type=str, default="none", help="Patologie pregresse/concomitanti (separate da virgola)")
    parser.add_argument("--treat", type=str, required=True, help="Patologie target da curare (separate da virgola)")

    args = parser.parse_args()

    existing_conditions = [c.strip() for c in args.conditions.split(',')]
    diseases_to_treat = [d.strip() for d in args.treat.split(',')]

    # FIX: Passaggio dell'intero quadro clinico polipatologico
    patient_profile = {
        'age': args.age,
        'weight': args.weight,
        'sex': args.sex,
        'concomitant': existing_conditions 
    }

    print("\n" + "="*70)
    print(" 🏥 SAFETHERAPY AI - Multi-Target Optimization Engine")
    print("="*70)
    print(f" [👤] Paziente : {args.age} anni, {args.weight} kg, Sesso: {args.sex}")
    print(f" [📋] Anamnesi : {', '.join(existing_conditions)}")
    print(f" [🎯] Target   : {', '.join(diseases_to_treat)}")
    print("-" * 70)

    optimizer = TherapyOptimizer()
    solution_node = optimizer.solve(patient_profile, diseases_to_treat)

    print("\n" + "="*70)
    if solution_node:
        print(" ✅ PIANO TERAPEUTICO OTTIMIZZATO (SET COVER TROVATO)")
        print("="*70)
        print(f" {'FARMACO SCELTO':<25} | {'PATOLOGIE COPERTE (MULTI-TARGET)':<40}")
        print("-" * 70)
        
        for drug_atom, disease_atoms in solution_node.selected_drugs.items():
            real_drug_name = optimizer.ai._get_original_name(drug_atom)
            real_diseases_str = format_disease_names(optimizer, disease_atoms)
            multi_target_flag = " ⭐ [MULTI]" if len(disease_atoms) > 1 else ""
            print(f" {real_drug_name[:25]:<25} | {real_diseases_str}{multi_target_flag}")
            
        print("-" * 70)
        print(f" 📊 Score di Rischio/Costo (Minimizzato): {solution_node.f:.2f}")
        print(f"    (Penalità Farmacologica g(n): {solution_node.g:.2f})")
        print("="*70 + "\n")
    else:
        print(" ❌ NESSUNA TERAPIA SICURA TROVATA.")
        print("    L'agente non è riuscito a trovare una combinazione che soddisfi")
        print("    i vincoli di sicurezza (hard constraints) per tutte le malattie.")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()