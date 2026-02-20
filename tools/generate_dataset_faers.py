import pandas as pd
import numpy as np
import requests
import time
import os
import random
from tqdm import tqdm

class FaersMiningBot:
    def __init__(self, cases_per_drug=150):
        """
        cases_per_drug: Numero di casi REALI da cercare per OGNI singolo farmaco.
        Il numero finale di righe sarà (cases_per_drug * 2) * numero_farmaci.
        """
        self.cases_per_drug = cases_per_drug
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.who_file = os.path.join(self.data_dir, "WHO ATC-DDD 2024-07-31.csv")
        self.output_file = os.path.join(self.data_dir, "faers_smart_dataset.csv")
        
        self.api_url = "https://api.fda.gov/drug/event.json"
        self.api_key = None # Inserisci qui la tua API Key se ne hai una, per andare ancora più veloce

        # Parole chiave per mappare le malattie comuni in modo consistente
        self.keywords = {
            'diabetes': 'Diabetes', 'hypertens': 'Hypertension', 'pressure': 'Hypertension',
            'cardiac': 'Heart Disease', 'heart': 'Heart Disease', 'pain': 'Chronic Pain',
            'arthri': 'Arthritis', 'cholesterol': 'High Cholesterol', 'lipid': 'High Cholesterol',
            'depression': 'Depression', 'anxiety': 'Anxiety'
        }

    def load_who_drugs(self):
        print(f"[INIT] Caricamento whitelist da: {self.who_file} ...")
        if not os.path.exists(self.who_file):
            raise FileNotFoundError(f"❌ ERRORE: File {self.who_file} non trovato!")

        try:
            df = pd.read_csv(self.who_file)
            target_col = None
            for col in df.columns:
                if 'name' in col.lower() and 'atc' in col.lower():
                    target_col = col
                    break
            if not target_col: target_col = df.columns[1]
            
            drugs = df[target_col].dropna().unique().tolist()
            drugs = [str(d).strip().lower() for d in drugs if len(str(d)) > 3]
            return list(set(drugs))
        except Exception as e:
            print(f"❌ Errore lettura WHO: {e}")
            return []

    def generate_smart_shadows(self, df_real):
        """
        Genera i casi SICURI (TARGET=0). Invece di copiare i malati, crea
        profili di pazienti standard (media età, peso forma, meno patologie)
        per dare al ML un contrasto matematico netto.
        """
        safe_data = []
        for _, row in df_real.iterrows():
            # 1. Età realistica ma normalizzata (es. tra i 20 e i 65)
            age = np.random.normal(45, 12)
            age = max(18, min(75, age))
            
            # 2. Sesso bilanciato e Peso Forma
            sex = random.choice(['M', 'F'])
            if sex == 'M': weight = np.random.normal(80, 10)
            else: weight = np.random.normal(62, 8)
            weight = max(45, min(110, weight))
            
            # 3. Comorbidità: Il 70% delle volte il paziente sano non ha comorbidità, 
            # altrimenti ne ha una casuale e DIVERSA da quella del malato.
            if random.random() < 0.70:
                concomitant = 'None'
            else:
                conditions = list(set(self.keywords.values()))
                if row['CONCOMITANT'] in conditions:
                    conditions.remove(row['CONCOMITANT'])
                concomitant = random.choice(conditions) if conditions else 'None'

            safe_data.append({
                'AGE': int(age),
                'SEX': sex,
                'WEIGHT': round(weight, 1),
                'DRUG_NAME': row['DRUG_NAME'],
                'CONCOMITANT': concomitant,
                'REACTION_DESC': 'No Adverse Event',
                'TARGET': 0
            })
        return pd.DataFrame(safe_data)

    def _extract_consistent(self, report, drug_name):
        """Estrae un singolo record pulito dal JSON della FDA."""
        try:
            p = report.get('patient', {})
            
            # Età
            age = p.get('patientonsetage')
            if not age or p.get('patientonsetageunit') != '801': return None
            age = int(float(age))
            if age < 0 or age > 115: return None
            
            # Sesso
            sex_code = p.get('patientsex')
            if sex_code == '1': sex = 'M'
            elif sex_code == '2': sex = 'F'
            else: return None
            
            # Peso
            weight = p.get('patientweight')
            if weight:
                weight = float(weight)
                if weight < 5 or weight > 250: return None
            else:
                base = 78.0 if sex == 'M' else 65.0
                weight = base + np.random.normal(0, 8)
            
            # Comorbidità
            concomitant = 'None'
            for d in p.get('drug', []):
                indication = str(d.get('drugindication', '')).lower()
                for key, label in self.keywords.items():
                    if key in indication:
                        concomitant = label
                        break
                if concomitant != 'None': break
            
            reaction = 'Unknown'
            if p.get('reaction'):
                reaction = p['reaction'][0].get('reactionmeddrapt', 'Unknown')

            return {
                'AGE': age,
                'SEX': sex,
                'WEIGHT': round(weight, 1),
                'DRUG_NAME': drug_name,
                'CONCOMITANT': concomitant,
                'REACTION_DESC': reaction,
                'TARGET': 1
            }
        except:
            return None

    def run(self):
        who_drugs = self.load_who_drugs()
        if not who_drugs:
            return
            
        print(f"\n--- START MINING: Target {self.cases_per_drug} casi per {len(who_drugs)} farmaci ---")
        
        # Inizializza/Pulisce il file di output
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            
        total_records_saved = 0
        pbar = tqdm(total=len(who_drugs), desc="Progresso Farmaci")

        for drug in who_drugs:
            # Chiamata API massiva per singolo farmaco (Velocità 10x)
            params = {
                'search': f'patient.drug.medicinalproduct:"{drug}"',
                'limit': min(1000, self.cases_per_drug * 3) # Chiediamo il triplo per scartare gli sporchi
            }
            if self.api_key: params['api_key'] = self.api_key
            
            real_cases = []
            
            try:
                resp = requests.get(self.api_url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    results = resp.json().get('results', [])
                    for report in results:
                        extracted = self._extract_consistent(report, drug)
                        if extracted:
                            real_cases.append(extracted)
                        if len(real_cases) >= self.cases_per_drug:
                            break
                            
                elif resp.status_code == 429:
                    # Rate Limit colpito
                    time.sleep(3)
                    pbar.update(0)
                    continue

            except Exception:
                pass # Ignora errori di connessione e passa al prossimo farmaco
                
            # Se abbiamo trovato casi reali per questo farmaco, generiamo gli shadow e salviamo
            if real_cases:
                df_real = pd.DataFrame(real_cases)
                df_shadow = self.generate_smart_shadows(df_real)
                
                # Uniamo e mischiamo per il chunk corrente
                df_chunk = pd.concat([df_real, df_shadow], ignore_index=True)
                df_chunk = df_chunk.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # SALVATAGGIO INCREMENTALE (Append mode)
                file_exists = os.path.isfile(self.output_file)
                df_chunk.to_csv(self.output_file, mode='a', header=not file_exists, index=False)
                
                total_records_saved += len(df_chunk)
            
            # Pausa di rispetto per l'API (FDA chiede max 40 req/min senza key = 1.5 sec)
            time.sleep(1.5 if not self.api_key else 0.5)
            pbar.update(1)
            
        pbar.close()
        print("\n" + "="*50)
        print(f"✅ DATASET COMPLETATO: {total_records_saved} righe salvate in {self.output_file}")

if __name__ == "__main__":
    # 150 casi per farmaco * 300 farmaci * 2 (shadow) = ~90.000 righe pulite e bilanciate!
    bot = FaersMiningBot(cases_per_drug=150)
    bot.run()