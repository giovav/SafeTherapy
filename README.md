# 🏥 SafeTherapy

SafeTherapy è un agente intelligente per il **supporto alla decisione clinica** in contesti di polifarmacia. Dato il profilo clinico di un paziente (età, peso, sesso, anamnesi) e le patologie da trattare, il sistema esplora automaticamente lo spazio delle combinazioni farmacologiche e restituisce il piano terapeutico ottimizzato: il numero minimo di farmaci che copra tutte le patologie, rispettando i vincoli di sicurezza e minimizzando il rischio stimato per quel paziente.

---

> ## 📚 Documentazione
> La documentazione completa del progetto — architettura, dataset, decisioni di progetto, snippet di codice e risultati di valutazione — è disponibile [qui](docs/Report.md).

---
## ⚙️ Come funziona

Il sistema integra quattro componenti di intelligenza artificiale in un'architettura neuro-simbolica:

- 🔷 **Knowledge Base Prolog** — ontologia farmacologica in logica del primo ordine. Classifica i farmaci per meccanismo d'azione, deduce le linee terapeutiche appropriate e rileva interazioni pericolose (DDI) tra farmaci. Opera come filtro di sicurezza assoluto: le sue conclusioni non sono negoziabili.

- 🌲 **Random Forest** — classificatore supervisionato addestrato sul dataset FAERS. Per ogni farmaco candidato stima la probabilità di reazione avversa dato il profilo del paziente, producendo una penalità morbida che orienta la ricerca.

- 🕸️ **Rete Bayesiana** — modello probabilistico che quantifica la fragilità sistemica del paziente in base a età, peso e comorbidità. Il valore di fragilità agisce come moltiplicatore globale del rischio: per pazienti anziani o polipatologici tutte le penalità vengono amplificate.

- ⭐ **Algoritmo A\*** — motore di ricerca informata che esplora lo spazio degli stati terapeutici minimizzando la funzione di costo `f(n) = g(n) + h(n)`, dove `g(n)` incorpora appropriatezza clinica, rischio ML e fragilità BN, e `h(n)` è un'euristica ammissibile per garantire l'ottimalità della soluzione.

---

## 📁 Struttura del Progetto

```
SafeTherapy/
│
├── data/
│   ├── WHO ATC-DDD 2024-07-31.csv      # Catalogo farmacologico WHO (6.030 principi attivi)
│   └── faers_smart_dataset.csv         # Dataset semi-sintetico FAERS (~618k campioni)
│
├── src/
│   ├── kb/                             # Knowledge Base (componente simbolica)
│   │   ├── fact_extractor.py           # ETL: genera l'A-Box Prolog dal catalogo WHO
│   │   ├── interface.py                # Bridge Python-Prolog (PySwip)
│   │   ├── utils.py                    # to_prolog_atom(), ProjectConfig, TextUtils
│   │   └── prolog/
│   │       ├── reasoning.pl            # T-Box: ontologia farmacodinamica, linee terapeutiche, DDI
│   │       ├── facts.pl                # A-Box generata automaticamente (has_atc_code/2)
│   │       └── atom_mapping.json       # Dizionario atomo Prolog → nome originale FAERS
│   │
│   ├── ml/                             # Machine Learning (Random Forest)
│   │   ├── train_model.py              # Training offline: Nested CV, GridSearch, serializzazione
│   │   ├── predictor.py                # Inferenza real-time: predict_risk()
│   │   └── models/
│   │       ├── rf_risk_model.pkl       # Modello Random Forest serializzato
│   │       └── label_encoders.pkl      # LabelEncoder per variabili categoriche
│   │
│   ├── bn/                             # Rete Bayesiana
│   │   ├── learner.py                  # Training offline: DAG, BDeu, Goodness-of-Fit
│   │   ├── predictor.py                # Inferenza real-time: get_patient_fragility()
│   │   └── models/
│   │       └── faers_frailty_bbn.pkl   # Rete Bayesiana serializzata
│   │
│   └── sss/                            # SafeTherapy Search System (A*)
│       ├── search.py                   # TherapyOptimizer: algoritmo A*, TherapyNode
│       └── heuristic.py                # AIHeuristic: g(n) e h(n) neuro-simbolici
│
├── docs/
│   ├── Report.md                       # Documentazione completa del progetto
│   ├── generate_pdf.sh                 # Script per generare il PDF dal Markdown
│   ├── plots/
│   │   ├── rf_fold_metrics.png         # Metriche RF per fold (Nested CV)
│   │   ├── rf_metrics_boxplot.png      # Boxplot distribuzione metriche RF
│   │   ├── rf_feature_importances.png  # Feature importances Random Forest
│   │   ├── bn_dag_structure.png        # Grafo DAG della Rete Bayesiana
│   │   └── bn_cpt_heatmaps.png         # Heatmap delle CPT (P(IsFragile=1))
│   └── metrics/
│       ├── rf_evaluation_report.json   # Report completo valutazione Random Forest
│       └── bn_model_report.json        # Report Goodness-of-Fit Rete Bayesiana
│
└── README.md
```

---

## 📦 Requisiti

```
python >= 3.10
swi-prolog >= 9.x
uv >= 0.4
```

Le dipendenze Python sono gestite da `uv` tramite `pyproject.toml`:

```
pandas, numpy, scikit-learn, pgmpy, pyswip, joblib, matplotlib, seaborn, networkx, tqdm
```

---

## 🚀 Avvio rapido

```bash
# 1. Installa le dipendenze
uv sync

# 2. Avvia SafeTherapy con il profilo del paziente
uv run python -m src.main --age <età> --weight <peso> --sex <M|F> \
                           --conditions "<condizione, ...>" \
                           --treat "<sintomo, ...>"
```

**Esempio:**

```bash
uv run python -m src.main --age 21 --weight 50 --sex M \
                           --conditions "hypertension" \
                           --treat "pain, headache"
```

> ℹ️ **I modelli addestrati e l'A-Box sono già inclusi nel repository**, quindi non è necessario alcun passaggio aggiuntivo prima dell'esecuzione.

---

## 🔧 Riaddestrare i modelli (opzionale)

Qualora si volessero rigenerare l'A-Box Prolog o riaddestrare i modelli da zero, è possibile eseguire i seguenti comandi prima dell'avvio:

```bash
# Rigenera l'A-Box Prolog dal catalogo WHO
uv run python src/kb/fact_extractor.py

# Riaddestra il Random Forest (può richiedere diversi minuti)
uv run python src/ml/train_model.py

# Riaddestra la Rete Bayesiana
uv run python src/bn/learner.py
```

---