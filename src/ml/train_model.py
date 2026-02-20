# File: src/ml/train_model.py

"""
Modulo di Addestramento (Offline Training).
Effettua l'ottimizzazione e il training del classificatore Random Forest
utilizzando una Nested Cross-Validation per prevenire data leakage e overfitting.

Output generati automaticamente in SafeTherapy/docs/:
  plots/rf_fold_metrics.png       — metriche per fold con medie
  plots/rf_metrics_boxplot.png    — boxplot distribuzione metriche
  plots/rf_feature_importances.png — importanze delle feature
  metrics/rf_evaluation_report.json — report completo di valutazione
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DPI = 150


class RiskModelTrainer:
    """
    Gestisce l'intero ciclo di vita dell'addestramento del modello predittivo,
    dall'ingestion del dataset fino alla serializzazione del modello, alla
    produzione del report JSON e alla generazione automatica dei grafici
    per la documentazione.

    Tutti gli output destinati alla documentazione vengono salvati sotto
    SafeTherapy/docs/, separati dal codice sorgente e dai modelli serializzati.

    Attributes:
        base_dir (str): Directory radice del progetto (SafeTherapy/).
        dataset_path (str): Percorso al dataset FAERS di training.
        model_dir (str): Directory per i file serializzati (src/ml/models/).
        docs_plots_dir (str): Directory per i grafici (docs/plots/).
        docs_metrics_dir (str): Directory per i report (docs/metrics/).
        model_path (str): Percorso al modello Random Forest serializzato.
        encoder_path (str): Percorso ai LabelEncoder serializzati.
        report_path (str): Percorso al report JSON di valutazione.
    """

    def __init__(self):
        """Inizializza i path per dataset, modelli e output per la documentazione."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dataset_path = os.path.join(self.base_dir, "data", "faers_smart_dataset.csv")

        self.model_dir      = os.path.join(self.base_dir, "src", "ml", "models")
        self.docs_plots_dir = os.path.join(self.base_dir, "docs", "plots")
        self.docs_metrics_dir = os.path.join(self.base_dir, "docs", "metrics")

        self.model_path   = os.path.join(self.model_dir, "rf_risk_model.pkl")
        self.encoder_path = os.path.join(self.model_dir, "label_encoders.pkl")
        self.report_path  = os.path.join(self.docs_metrics_dir, "rf_evaluation_report.json")

    # ------------------------------------------------------------------
    # GRAFICI
    # ------------------------------------------------------------------

    def _save_fold_metrics_plot(self, fold_results: list, summary: dict) -> None:
        """
        Genera e salva un grafico a barre raggruppate con le metriche per fold
        e linee tratteggiate che indicano la media complessiva.

        Args:
            fold_results (list[dict]): Lista dei risultati per fold.
            summary (dict): Dizionario con mean e std per ciascuna metrica.
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        folds   = [f"Fold {r['fold']}" for r in fold_results]
        colors  = sns.color_palette("muted", len(metrics))
        x       = np.arange(len(folds))
        width   = 0.2

        fig, ax = plt.subplots(figsize=(11, 5))

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [r[metric] for r in fold_results]
            ax.bar(x + i * width, values, width,
                   label=metric.capitalize(), color=color, alpha=0.85)
            mean_val = summary[metric]['mean']
            ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=1.2, alpha=0.7,
                       label=f"Media {metric.capitalize()} ({mean_val:.3f})")

        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Valore metrica", fontsize=12)
        ax.set_title("Random Forest — Metriche per Fold (Nested CV)", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(folds)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        fig.tight_layout()

        path = os.path.join(self.docs_plots_dir, "rf_fold_metrics.png")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"[ML-PLOT] Salvato: {path}")

    def _save_metrics_boxplot(self, fold_results: list) -> None:
        """
        Genera e salva un boxplot delle metriche sui fold per visualizzare
        la dispersione e identificare eventuali fold anomali.

        Args:
            fold_results (list[dict]): Lista dei risultati per fold.
        """
        metrics  = ['accuracy', 'precision', 'recall', 'f1']
        data     = {m.capitalize(): [r[m] for r in fold_results] for m in metrics}
        df_plot  = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(7, 5))
        df_plot.boxplot(
            ax=ax, patch_artist=True,
            boxprops=dict(facecolor='#AEC6CF', color='#333'),
            medianprops=dict(color='#E05252', linewidth=2)
        )
        ax.set_title("Random Forest — Distribuzione Metriche sui Fold",
                     fontsize=13, fontweight='bold')
        ax.set_ylabel("Valore metrica", fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        fig.tight_layout()

        path = os.path.join(self.docs_plots_dir, "rf_metrics_boxplot.png")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"[ML-PLOT] Salvato: {path}")

    def _save_feature_importance_plot(self, feature_importances: dict) -> None:
        """
        Genera e salva un grafico a barre orizzontali delle feature importances
        del modello finale, ordinate in modo decrescente.

        Args:
            feature_importances (dict): Dizionario feature → importanza,
                già ordinato in modo decrescente.
        """
        features = list(feature_importances.keys())
        values   = list(feature_importances.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(
            features[::-1], values[::-1],
            color=sns.color_palette("muted", len(features))[::-1],
            alpha=0.9
        )
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va='center', fontsize=10)

        ax.set_xlabel("Importanza (Gini)", fontsize=11)
        ax.set_title("Random Forest — Feature Importances (modello finale)",
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.25)
        fig.tight_layout()

        path = os.path.join(self.docs_plots_dir, "rf_feature_importances.png")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"[ML-PLOT] Salvato: {path}")

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Esegue la pipeline completa di Machine Learning.

        Fasi:
            1. Caricamento e encoding del dataset FAERS.
            2. Nested Cross-Validation (5 outer, 3 inner fold) con GridSearchCV.
            3. Calcolo di accuracy, precision, recall e F1 per ciascun fold.
            4. Refit finale sull'intero dataset.
            5. Salvataggio modello, encoder, report JSON e grafici.

        Returns:
            dict: Report completo di valutazione, identico a quello salvato
                su disco in docs/metrics/rf_evaluation_report.json.

        Raises:
            FileNotFoundError: Se il dataset non è presente nel percorso configurato.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"[ML-TRAINER] Impossibile trovare il dataset: {self.dataset_path}"
            )

        print(f"[ML-TRAINER] Ingestion dataset: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        feature_cols = ['AGE', 'SEX', 'WEIGHT', 'DRUG_NAME', 'CONCOMITANT']
        X = df[feature_cols].copy()
        y = df['TARGET']

        print(f"[ML-TRAINER] Dataset: {len(df)} campioni | "
              f"Positivi: {y.sum()} ({y.mean()*100:.1f}%) | "
              f"Negativi: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

        encoders = {}
        categorical_cols = ['SEX', 'DRUG_NAME', 'CONCOMITANT']

        print("[ML-TRAINER] Adattamento dei dizionari categoriali (Label Encoders)...")
        for col in categorical_cols:
            X[col] = X[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # class_weight='balanced' compensa lo sbilanciamento atteso nel dataset FAERS.
        # n_estimators, max_depth e min_samples_split bilanciano capacità
        # espressiva e regolarizzazione.
        param_grid = {
            'n_estimators':     [100, 200],
            'max_depth':        [10, 20, None],
            'min_samples_split':[2, 5, 10],
            'class_weight':     ['balanced']
        }
        rf = RandomForestClassifier(random_state=42)

        print("\n[ML-TRAINER] Avvio Nested Cross-Validation (5 outer fold, 3 inner fold)...")
        fold_results = []
        outer_acc, outer_prec, outer_rec, outer_f1 = [], [], [], []

        fold_iterator = tqdm(
            outer_cv.split(X, y),
            total=outer_cv.get_n_splits(),
            desc="Training Progress",
            unit="fold"
        )

        for fold, (train_idx, test_idx) in enumerate(fold_iterator):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_iterator.set_description(f"Ottimizzazione Fold {fold+1}/5")

            clf = GridSearchCV(
                estimator=rf, param_grid=param_grid,
                cv=inner_cv, scoring='f1', n_jobs=-1
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)

            outer_acc.append(acc);  outer_prec.append(prec)
            outer_rec.append(rec);  outer_f1.append(f1)

            best_params = {k: v for k, v in clf.best_params_.items() if k != 'class_weight'}
            fold_results.append({
                'fold': fold + 1,
                'accuracy':    round(acc,  4),
                'precision':   round(prec, 4),
                'recall':      round(rec,  4),
                'f1':          round(f1,   4),
                'best_params': best_params
            })

            tqdm.write(
                f"    ✓ [Fold {fold+1}] "
                f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | "
                f"Params: {best_params}"
            )

        summary = {
            'accuracy':  {'mean': round(np.mean(outer_acc),  4), 'std': round(np.std(outer_acc),  4)},
            'precision': {'mean': round(np.mean(outer_prec), 4), 'std': round(np.std(outer_prec), 4)},
            'recall':    {'mean': round(np.mean(outer_rec),  4), 'std': round(np.std(outer_rec),  4)},
            'f1':        {'mean': round(np.mean(outer_f1),   4), 'std': round(np.std(outer_f1),   4)},
        }

        print(f"\n[ML-TRAINER] === Valutazione Globale (media ± std su 5 fold) ===")
        for metric, vals in summary.items():
            print(f"    {metric:<12}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        print("\n[ML-TRAINER] Refit finale sull'intero dataset...")
        final_clf = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        with tqdm(total=1, desc="Final Refit",
                  bar_format="{l_bar}{bar}| [Elaborazione in corso...]") as pbar:
            final_clf.fit(X, y)
            pbar.update(1)

        best_final_params = {
            k: v for k, v in final_clf.best_params_.items() if k != 'class_weight'
        }
        print(f"[ML-TRAINER] Iperparametri finali: {best_final_params}")

        importances = final_clf.best_estimator_.feature_importances_
        feature_importance_dict = {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(feature_cols, importances),
                key=lambda x: x[1], reverse=True
            )
        }
        print("[ML-TRAINER] Feature importances:")
        for feat, imp in feature_importance_dict.items():
            print(f"    {feat:<20}: {imp:.4f}")

        # ── Salvataggio artefatti ──────────────────────────────────────
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.docs_plots_dir, exist_ok=True)
        os.makedirs(self.docs_metrics_dir, exist_ok=True)

        joblib.dump(final_clf.best_estimator_, self.model_path)
        joblib.dump(encoders, self.encoder_path)

        report = {
            'model': 'RandomForestClassifier',
            'validation_strategy': 'Nested Cross-Validation (5 outer, 3 inner)',
            'scoring_metric': 'F1',
            'dataset': {
                'size': len(df),
                'positive_samples': int(y.sum()),
                'negative_samples': int((~y.astype(bool)).sum()),
                'positive_rate': round(float(y.mean()), 4)
            },
            'per_fold_results': fold_results,
            'summary': summary,
            'best_params_final': best_final_params,
            'feature_importances': feature_importance_dict
        }

        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n[ML-TRAINER] Generazione grafici per la documentazione...")
        self._save_fold_metrics_plot(fold_results, summary)
        self._save_metrics_boxplot(fold_results)
        self._save_feature_importance_plot(feature_importance_dict)

        print(f"\n[ML-TRAINER] Modello salvato      : {self.model_path}")
        print(f"[ML-TRAINER] Encoders salvati     : {self.encoder_path}")
        print(f"[ML-TRAINER] Report salvato       : {self.report_path}")
        print(f"[ML-TRAINER] Grafici salvati in   : {self.docs_plots_dir}")

        return report


if __name__ == "__main__":
    trainer = RiskModelTrainer()
    trainer.train()