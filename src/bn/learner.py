# File: src/bn/learner.py

import os
import json
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, BDeu, BIC

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DPI = 150


class FaersBN:
    """
    Rete Bayesiana Discreta: Modulo di Addestramento (Offline).

    Gestisce la creazione della Rete Bayesiana addestrata sul dataset FAERS.
    Esegue la costruzione del grafo, l'apprendimento delle CPT (Conditional 
    Probability Tables) tramite BDeu estimator, ed estrae metriche di 
    Goodness-of-Fit (BIC e BDeu) per valutarne la robustezza strutturale.
    Genera inoltre una rappresentazione grafica del DAG e delle CPT.
    
    Questo modulo deve essere eseguito in modalità standalone.

    Attributes:
        base_dir (str): Directory radice del progetto.
        model_dir (str): Directory per i file serializzati.
        docs_plots_dir (str): Directory per i grafici.
        docs_metrics_dir (str): Directory per i report JSON.
        model_path (str): Percorso di salvataggio del modello `.pkl`.
        report_path (str): Percorso di salvataggio del report di validazione.
        dataset_path (str): Percorso al dataset FAERS per il training.
        network (DiscreteBayesianNetwork): L'istanza del grafo in addestramento.
    """

    def __init__(self):
        """Inizializza i percorsi di sistema per input (dati) e output (modelli e docs)."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir        = os.path.join(self.base_dir, "src", "bn", "models")
        self.docs_plots_dir   = os.path.join(self.base_dir, "docs", "plots")
        self.docs_metrics_dir = os.path.join(self.base_dir, "docs", "metrics")

        self.model_path  = os.path.join(self.model_dir, "faers_frailty_bbn.pkl")
        self.report_path = os.path.join(self.docs_metrics_dir, "bn_model_report.json")
        self.dataset_path = os.path.join(self.base_dir, "data", "faers_smart_dataset.csv")

        self.network = None

    def _discretize_age(self, age: float) -> str:
        """
        Discretizza l'età continua in una categoria clinica ordinale.
        
        Args:
            age (float): Età del paziente in anni.
            
        Returns:
            str: Categoria clinica ('pediatric', 'adult', 'geriatric').
        """
        if age < 18: return 'pediatric'
        elif age < 65: return 'adult'
        else: return 'geriatric'

    def _discretize_weight(self, weight: float) -> str:
        """
        Discretizza il peso corporeo in una categoria metabolica.
        
        Args:
            weight (float): Peso in kg.
            
        Returns:
            str: Categoria metabolica ('underweight', 'normal', 'overweight').
        """
        if weight < 50: return 'underweight'
        elif weight <= 90: return 'normal'
        else: return 'overweight'

    def _build_train_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Applica la pipeline di discretizzazione a un DataFrame grezzo FAERS.
        
        Args:
            df_raw (pd.DataFrame): DataFrame contenente i dati non processati.
            
        Returns:
            pd.DataFrame: Dati discretizzati pronti per il fit della Rete Bayesiana.
        """
        df_out = pd.DataFrame()
        df_out['AgeGroup'] = df_raw['AGE'].apply(self._discretize_age)
        df_out['WeightGroup'] = df_raw['WEIGHT'].apply(self._discretize_weight)
        df_out['HasConcomitant'] = df_raw['CONCOMITANT'].apply(
            lambda x: "0" if pd.isna(x) or str(x).strip().lower() == 'none' else "1"
        )
        df_out['IsFragile'] = df_raw['TARGET'].astype(str)
        for col in df_out.columns:
            df_out[col] = df_out[col].astype('category')
        return df_out

    def _save_dag_plot(self) -> None:
        """
        Disegna e salva il Directed Acyclic Graph (DAG) della rete.
        Utilizza posizionamento gerarchico esplicito per prevenire la sovrapposizione
        dei nodi che causa il bug di StopIteration su matplotlib.
        """
        if self.network is None:
            return

        plt.figure(figsize=(10, 6))
        
        # Posizionamento fisso: Genitori in alto (y=1), Figlio in basso (y=0)
        pos = {
            'AgeGroup': (0, 1),
            'WeightGroup': (1, 1),
            'HasConcomitant': (2, 1),
            'IsFragile': (1, 0)
        }
        
        nx.draw(
            self.network, pos, 
            with_labels=True, 
            node_size=3500, 
            node_color="#AEC6CF", 
            font_size=11, 
            font_weight="bold", 
            edge_color="gray",
            arrows=True, 
            arrowsize=20
        )
        
        plt.title("Rete Bayesiana — Struttura Causale (DAG)", fontsize=14, fontweight='bold')
        plt.margins(0.3)
        plt.tight_layout()
        
        path = os.path.join(self.docs_plots_dir, "bn_dag_structure.png")
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"[BBN-PLOT] Salvato DAG in: {path}")

    def _save_cpt_heatmaps(self) -> None:
        """
        Genera e salva le heatmap delle Conditional Probability Tables (CPT)
        apprese dalla Rete Bayesiana. Appiattisce i tensori N-Dimensionali di
        pgmpy per la ricostruzione matriciale 3D in NumPy.
        """
        if self.network is None:
            print("[BBN-PLOT] Rete non addestrata, CPT non disponibili.")
            return

        try:
            cpd_fragile = self.network.get_cpds('IsFragile')
        except Exception as e:
            print(f"[BBN-PLOT] Impossibile recuperare CPT IsFragile: {e}")
            return

        fragile_states = cpd_fragile.state_names['IsFragile']
        pos_idx = fragile_states.index('1')
        
        cpt_values = cpd_fragile.values[pos_idx].flatten()

        age_states  = ['pediatric', 'adult', 'geriatric']
        wgt_states  = ['underweight', 'normal', 'overweight']
        conc_states = ['0', '1']
        conc_labels = ['Assente', 'Presente']

        idx = 0
        cpt_matrix = np.zeros((len(age_states), len(wgt_states), len(conc_states)))
        for ai in range(len(age_states)):
            for wi in range(len(wgt_states)):
                for ci in range(len(conc_states)):
                    cpt_matrix[ai, wi, ci] = cpt_values[idx]
                    idx += 1

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Rete Bayesiana — P(IsFragile=1 | Genitore)\n(marginalizzato sugli altri genitori)", fontsize=13, fontweight='bold')

        age_means  = cpt_matrix.mean(axis=(1, 2)).reshape(1, -1)
        wgt_means  = cpt_matrix.mean(axis=(0, 2)).reshape(1, -1)
        conc_means = cpt_matrix.mean(axis=(0, 1)).reshape(1, -1)

        for ax, data, xlabels, title in [
            (axes[0], age_means,  age_states,  "AgeGroup"),
            (axes[1], wgt_means,  wgt_states,  "WeightGroup"),
            (axes[2], conc_means, conc_labels, "HasConcomitant"),
        ]:
            sns.heatmap(data, ax=ax, annot=True, fmt=".3f", xticklabels=xlabels, yticklabels=["P(IsFragile=1)"], cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
            ax.set_title(title, fontsize=11)
            ax.set_xticklabels(xlabels, rotation=15)

        fig.tight_layout()
        path = os.path.join(self.docs_plots_dir, "bn_cpt_heatmaps.png")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"[BBN-PLOT] Salvate Heatmap in: {path}")

    def train_and_save(self) -> None:
        """
        Addestra l'architettura Bayesiana, ne calcola le statistiche di
        Goodness-of-Fit e serializza il modello, i report e i grafici.
        """
        if not os.path.exists(self.dataset_path):
            print(f"❌ [BBN-ERROR] Dataset FAERS non trovato in {self.dataset_path}.")
            return

        print(f"[BBN-LEARN] Caricamento dataset FAERS unificato da: {self.dataset_path}")
        df_raw = pd.read_csv(self.dataset_path)
        df_raw = df_raw.dropna(subset=['AGE', 'WEIGHT', 'TARGET'])

        print("[BBN-LEARN] Discretizzazione delle variabili biomediche in corso...")
        df_train = self._build_train_dataframe(df_raw)

        print("[BBN-LEARN] Costruzione del Grafo Bayesiano Diretto (DAG)...")
        edges = [('AgeGroup', 'IsFragile'), ('WeightGroup', 'IsFragile'), ('HasConcomitant', 'IsFragile')]
        self.network = DiscreteBayesianNetwork(edges)

        print("[BBN-LEARN] Apprendimento delle distribuzioni probabilistiche (CPT)...")
        self.network.fit(data=df_train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

        # Calcolo delle metriche di Goodness of Fit
        print("[BBN-LEARN] Calcolo statistiche di Goodness-of-Fit (BIC, BDeu)...")
        bdeu_score = BDeu(df_train, equivalent_sample_size=10).score(self.network)
        bic_score = BIC(df_train).score(self.network)
        
        print(f"    ✓ BDeu Score: {bdeu_score:.4f}")
        print(f"    ✓ BIC Score:  {bic_score:.4f}")

        # Serializzazione modello
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.docs_plots_dir, exist_ok=True)
        os.makedirs(self.docs_metrics_dir, exist_ok=True)
        
        joblib.dump({'network': self.network}, self.model_path)
        print(f"✅ [BBN-LEARN] Modello salvato in: {self.model_path}")

        # Generazione Grafici
        print("[BBN-LEARN] Generazione output visivi per la documentazione...")
        self._save_dag_plot()
        self._save_cpt_heatmaps()
        
        # Generazione Report
        report = {
            'model': 'DiscreteBayesianNetwork (pgmpy)',
            'structure': {
                'nodes': list(self.network.nodes()),
                'edges': list(self.network.edges())
            },
            'estimator': 'BayesianEstimator (BDeu, equivalent_sample_size=10)',
            'training_samples': len(df_train),
            'goodness_of_fit_metrics': {
                'bdeu_score': round(bdeu_score, 4),
                'bic_score': round(bic_score, 4)
            }
        }
        
        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"[BBN-LEARN] Report JSON salvato in: {self.report_path}")

if __name__ == "__main__":
    bbn = FaersBN()
    bbn.train_and_save()