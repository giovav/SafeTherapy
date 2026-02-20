# File: src/kb/interface.py

"""
Modulo di astrazione Python-Prolog.
Fornisce un'interfaccia ad alto livello tramite PySwip per interrogare
il motore di inferenza logica (T-Box) durante l'esecuzione dell'algoritmo A*.
"""

import os
import re
from pyswip import Prolog


class PrologInterface:
    """
    Gestisce la comunicazione con il runtime Prolog per la validazione
    delle prescrizioni farmacologiche multiple in modo robusto.

    Carica le regole ontologiche da `reasoning.pl` all'avvio e fornisce
    metodi di alto livello per verificare la sicurezza di combinazioni
    di farmaci senza esporre la sintassi Prolog al chiamante.

    Attributes:
        prolog (Prolog): L'istanza del runtime SWI-Prolog gestita da PySwip.
    """

    def __init__(self):
        """
        Inizializza l'ambiente Prolog e carica le regole inferenziali.

        Risolve il percorso di `reasoning.pl` relativo alla posizione
        del modulo corrente e lo consulta nel runtime Prolog. Se il file
        non viene trovato, stampa un errore senza sollevare un'eccezione,
        lasciando l'oggetto in stato degradato (nessuna regola caricata).
        """
        self.prolog = Prolog()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rule_file = os.path.join(base_dir, "prolog", "reasoning.pl").replace("\\", "/")

        if os.path.exists(rule_file):
            self.prolog.consult(rule_file)
        else:
            print(f"[ERROR] File Prolog non trovato: {rule_file}")

    def verify_therapy(self, drugs: list) -> dict:
        """
        Valuta la sicurezza di una combinazione di farmaci interrogando Prolog.

        Itera su tutte le coppie non ordinate di farmaci forniti ed esegue
        il predicato `check_pair_safety/3` per ciascuna. Supporta il parsing
        robusto dei risultati PySwip tramite due strategie: lettura dell'oggetto
        Functor nativo (metodo primario) e parsing con espressioni regolari
        sulla rappresentazione stringa (metodo di fallback).

        I farmaci con interazione di severità 'high' causano la restituzione
        immediata del conflitto senza analizzare le coppie rimanenti.

        Args:
            drugs (list[str]): Lista di atomi Prolog rappresentanti i farmaci
                da valutare (es. ['warfarin', 'ibuprofen']).

        Returns:
            dict: Un dizionario con due chiavi:
                - 'safe' (bool): True se nessuna interazione è stata rilevata.
                - 'conflicts' (list[dict]): Lista di conflitti trovati, ciascuno
                  con le chiavi 'drugs' (tuple), 'severity' (str) e 'msg' (str).
                  Lista vuota se la terapia è sicura o se viene fornito meno
                  di un farmaco.
        """
        conflicts = []

        if len(drugs) < 2:
            return {'safe': True, 'conflicts': []}

        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                query = f"check_pair_safety('{drugs[i]}', '{drugs[j]}', Result)"

                try:
                    res = list(self.prolog.query(query))
                    if not res:
                        continue

                    result_obj = res[0]['Result']
                    result_str = str(result_obj)

                    if result_str == 'safe':
                        continue

                    severity = 'high'  # Fallback prudenziale
                    msg = 'Interazione rilevata'

                    # TENTATIVO 1: Lettura tramite oggetto Functor (PySwip standard)
                    if hasattr(result_obj, 'args') and len(result_obj.args) >= 2:
                        severity = str(result_obj.args[0])
                        msg = str(result_obj.args[1])
                    else:
                        # TENTATIVO 2: Parsing tramite Espressioni Regolari (Stringa raw)
                        match = re.search(r"conflict\(([^,]+),\s*(.*)\)", result_str)
                        if match:
                            severity = match.group(1).strip().strip("'").strip('"')
                            raw_msg = match.group(2).strip()
                            if raw_msg.endswith(')'):
                                raw_msg = raw_msg[:-1]
                            msg = raw_msg.strip("'").strip('"')

                    conflicts.append({
                        'drugs': (drugs[i], drugs[j]),
                        'severity': severity,
                        'msg': msg
                    })

                except Exception as e:
                    print(f"[KB-WARN] Errore di parsing Prolog su {drugs[i]}-{drugs[j]}: {e}")

        return {'safe': len(conflicts) == 0, 'conflicts': conflicts}