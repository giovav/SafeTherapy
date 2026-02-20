% File: src/kb/prolog/reasoning.pl
%
% SafeTherapy - T-Box: Knowledge Base Intensionale
%
% Questo modulo implementa la componente intensionale della Knowledge Base
% del sistema SafeTherapy. Ragiona per deduzione a partire da proprietà
% farmacodinamiche e fisiopatologiche, senza enumerare esplicitamente
% le combinazioni farmaco-malattia.
%
% Struttura:
%   1. Ontologia farmacodinamica  — classificazione dei farmaci per meccanismo d'azione
%   2. Ontologia fisiopatologica  — classificazione delle malattie per bisogno terapeutico
%   3. Motore di inferenza        — deduzione delle linee terapeutiche appropriate
%   4. Calcolo dei costi          — assegnazione dei costi per l'algoritmo A*
%   5. Safety checker             — rilevazione delle interazioni farmacologiche (DDI)
%
% Dipendenze:
%   - facts.pl: A-Box estensionale con i fatti has_atc_code/2 generati dal WHO ATC-DDD
%
% Formato dei codici ATC:
%   I codici ATC sono gerarchici su 5 livelli (es. C09AA01).
%   Le regole usano sub_atom/5 per classificare i farmaci a livello
%   di sottogruppo terapeutico (caratteri 0-3 o 0-4), senza enumerare
%   i singoli principi attivi.

:- use_module(library(lists)).
:- consult('facts.pl').

:- discontiguous dangerous_classes/4.
:- discontiguous check_pair_safety/3.
:- discontiguous approved_for/3.
:- discontiguous disease_specific_cost/3.
:- discontiguous increases_bp/1.
:- discontiguous decreases_bp/1.

% ======================================================================
% 1. ONTOLOGIA FARMACODINAMICA
%
% Ogni predicato is_X(Drug) classifica un farmaco in una categoria
% terapeutica in base al prefisso del suo codice ATC.
% La classificazione è universalmente quantificata: vale per qualsiasi
% farmaco presente nell'A-Box che soddisfi il pattern ATC.
% ======================================================================

is_antacid(Drug)                 :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'a02a').
is_acid_suppressor(Drug)         :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'a02b').
is_insulin(Drug)                 :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'a10a').
is_oral_hypoglycemic(Drug)       :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'a10b').
is_antithrombotic(Drug)          :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'b01a').
is_antihypertensive(Drug)        :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c02').
is_diuretic(Drug)                :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c03').
is_beta_blocker(Drug)            :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c07').
is_calcium_channel_blocker(Drug) :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c08').
is_ace_inhibitor(Drug)           :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'c09a').
is_arb(Drug)                     :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'c09c').
is_lipid_lowering(Drug)          :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c10').
is_corticosteroid(Drug)          :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'h02').
is_thyroid_therapy(Drug)         :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'h03').
is_antibacterial(Drug)           :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'j01').
is_nsaid(Drug)                   :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'm01a').
is_opioid(Drug)                  :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'n02a').
is_analgesic_other(Drug)         :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'n02b').
is_antiepileptic(Drug)           :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'n03').
is_antipsychotic(Drug)           :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'n05a').
is_anxiolytic(Drug)              :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'n05b').
is_antidepressant(Drug)          :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'n06a').
is_bronchodilator(Drug)          :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'r03').
is_antihistamine(Drug)           :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 4, _, 'r06a').

% Categorie aggregate: un farmaco appartiene alla categoria se soddisfa
% almeno una delle sottocategorie che la compongono.
% Ogni alternativa è espressa come clausola separata (forma Horn definita).

is_analgesic(Drug) :- is_opioid(Drug).
is_analgesic(Drug) :- is_analgesic_other(Drug).
is_analgesic(Drug) :- is_nsaid(Drug).

is_bp_lowering(Drug) :- is_antihypertensive(Drug).
is_bp_lowering(Drug) :- is_ace_inhibitor(Drug).
is_bp_lowering(Drug) :- is_arb(Drug).
is_bp_lowering(Drug) :- is_beta_blocker(Drug).
is_bp_lowering(Drug) :- is_calcium_channel_blocker(Drug).
is_bp_lowering(Drug) :- is_diuretic(Drug).

is_anti_inflammatory(Drug) :- is_nsaid(Drug).
is_anti_inflammatory(Drug) :- is_corticosteroid(Drug).

% ======================================================================
% 2. ONTOLOGIA FISIOPATOLOGICA
%
% Ogni fatto requires_X(Disease) dichiara quale meccanismo d'azione
% terapeutico è necessario per trattare una determinata patologia.
% Questi fatti costituiscono il ponte logico tra le malattie (dominio
% clinico) e i farmaci (dominio farmacologico).
% ======================================================================

requires_analgesia(pain).
requires_analgesia(headache).
requires_analgesia(chest_pain).
requires_anti_inflammatory(arthritis).
requires_anti_inflammatory(gout).

requires_acid_suppression(gastroesophageal_reflux).
requires_acid_suppression(stomach_ulcer).
requires_acid_suppression(heartburn).
requires_acid_suppression(dyspepsia).

requires_bp_lowering(hypertension).
requires_diuresis(edema).
requires_diuresis(heart_failure).
requires_heart_rate_control(atrial_fibrillation).
requires_heart_rate_control(tachycardia).
requires_vasodilation(angina_pectoris).
requires_lipid_lowering(hypercholesterolemia).
requires_anticoagulation(pulmonary_embolism).

requires_antibacterial(pneumonia_bacterial).
requires_antibacterial(urinary_tract_infections).
requires_antibacterial(sepsis).
requires_antibacterial(staphylococcal_infections).
requires_antibacterial(escherichia_coli_infections).

requires_anxiolysis(anxiety_disorders).
requires_anxiolysis(panic_disorder).
requires_antidepressant(depression).
requires_antipsychotic(schizophrenia).
requires_seizure_control(epilepsy).

requires_bronchodilation(asthma).
requires_bronchodilation(bronchitis).
requires_antihistamine(rhinitis_allergic).
requires_antihistamine(urticaria).

requires_glucose_lowering(diabetes_mellitus_type_2).
requires_insulin(diabetes_mellitus_type_1).
requires_thyroid_regulation(hypothyroidism).

% ======================================================================
% 3. MOTORE DI INFERENZA: LINEE TERAPEUTICHE
%
% approved_for(Drug, Disease, Line) è derivato per deduzione:
% un farmaco è appropriato per una malattia se il suo meccanismo d'azione
% soddisfa il bisogno fisiopatologico della malattia stessa.
%
% Le linee terapeutiche riflettono le raccomandazioni cliniche:
%   Linea 1 — trattamento di prima scelta (efficacia massima, profilo
%             di sicurezza favorevole per quella indicazione)
%   Linea 2 — alternativa o trattamento palliativo (seconda scelta)
%   Linea 3 — estrema ratio (es. oppioidi per il dolore refrattario)
% ======================================================================

approved_for(Drug, Disease, 1) :- requires_analgesia(Disease),          is_analgesic_other(Drug).
approved_for(Drug, Disease, 1) :- requires_anti_inflammatory(Disease),  is_anti_inflammatory(Drug).
approved_for(Drug, Disease, 1) :- requires_acid_suppression(Disease),   is_acid_suppressor(Drug).
approved_for(Drug, Disease, 1) :- requires_bp_lowering(Disease),        is_ace_inhibitor(Drug).
approved_for(Drug, Disease, 1) :- requires_bp_lowering(Disease),        is_arb(Drug).
approved_for(Drug, Disease, 1) :- requires_diuresis(Disease),           is_diuretic(Drug).
approved_for(Drug, Disease, 1) :- requires_heart_rate_control(Disease), is_beta_blocker(Drug).
approved_for(Drug, Disease, 1) :- requires_vasodilation(Disease),       is_calcium_channel_blocker(Drug).
approved_for(Drug, Disease, 1) :- requires_lipid_lowering(Disease),     is_lipid_lowering(Drug).
approved_for(Drug, Disease, 1) :- requires_anticoagulation(Disease),    is_antithrombotic(Drug).
approved_for(Drug, Disease, 1) :- requires_antibacterial(Disease),      is_antibacterial(Drug).
approved_for(Drug, Disease, 1) :- requires_anxiolysis(Disease),         is_anxiolytic(Drug).
approved_for(Drug, Disease, 1) :- requires_antidepressant(Disease),     is_antidepressant(Drug).
approved_for(Drug, Disease, 1) :- requires_antipsychotic(Disease),      is_antipsychotic(Drug).
approved_for(Drug, Disease, 1) :- requires_seizure_control(Disease),    is_antiepileptic(Drug).
approved_for(Drug, Disease, 1) :- requires_bronchodilation(Disease),    is_bronchodilator(Drug).
approved_for(Drug, Disease, 1) :- requires_antihistamine(Disease),      is_antihistamine(Drug).
approved_for(Drug, Disease, 1) :- requires_glucose_lowering(Disease),   is_oral_hypoglycemic(Drug).
approved_for(Drug, Disease, 1) :- requires_insulin(Disease),            is_insulin(Drug).
approved_for(Drug, Disease, 1) :- requires_thyroid_regulation(Disease), is_thyroid_therapy(Drug).

approved_for(Drug, Disease, 2) :- requires_analgesia(Disease),          is_nsaid(Drug).
approved_for(Drug, anxiety_disorders, 2)  :- is_beta_blocker(Drug).
approved_for(Drug, Disease, 2) :- requires_bp_lowering(Disease),        is_beta_blocker(Drug).
approved_for(Drug, asthma, 2)             :- is_corticosteroid(Drug).
approved_for(Drug, rhinitis_allergic, 2)  :- is_corticosteroid(Drug).
approved_for(Drug, Disease, 2) :- requires_acid_suppression(Disease),   is_antacid(Drug).

approved_for(Drug, pain, 3) :- is_opioid(Drug).

% ======================================================================
% 4. CALCOLO DEI COSTI PER L'ALGORITMO A*
%
% disease_specific_cost(Drug, Disease, Cost) assegna il costo
% prescrittivo in base alla linea terapeutica dedotta da approved_for/3.
%
% I valori riflettono la priorità clinica:
%   10   — prima linea  (costo minimo, scelta ottimale)
%   50   — seconda linea
%   200  — terza linea
%   2000 — farmaco non approvato per quella indicazione (off-label)
%
% La negazione per fallimento (\+) garantisce la mutua esclusione
% tra le clausole: ogni farmaco riceve esattamente un costo,
% corrispondente alla linea più bassa per cui è approvato.
% ======================================================================

disease_specific_cost(Drug, Disease, 10) :-
    approved_for(Drug, Disease, 1).

disease_specific_cost(Drug, Disease, 50) :-
    \+ approved_for(Drug, Disease, 1),
    approved_for(Drug, Disease, 2).

disease_specific_cost(Drug, Disease, 200) :-
    \+ approved_for(Drug, Disease, 1),
    \+ approved_for(Drug, Disease, 2),
    approved_for(Drug, Disease, 3).

disease_specific_cost(Drug, Disease, 2000) :-
    \+ approved_for(Drug, Disease, 1),
    \+ approved_for(Drug, Disease, 2),
    \+ approved_for(Drug, Disease, 3).

% ======================================================================
% 5. SAFETY CHECKER: INTERAZIONI FARMACOLOGICHE (DDI)
%
% check_pair_safety(Drug1, Drug2, Result) valuta la sicurezza di una
% coppia di farmaci. Result è unificato con:
%   safe                    — nessuna interazione rilevata
%   conflict(Severity, Msg) — interazione rilevata, con:
%       Severity = 'high'   → controindicazione assoluta
%       Severity = 'medium' → interazione clinicamente rilevante
%
% L'ordinamento delle clausole definisce la priorità di rilevazione:
%   1. Antagonismo fisiopatologico (effetti opposti sulla pressione)
%   2. DDI da classi pericolose note (dangerous_classes/4)
%   3. Cascata prescrittiva (FANS → ACE-inibitore)
%   4. Duplicazione terapeutica (stessa sottoclasse ATC)
%   5. Fallback: safe
%
% Ogni categoria è espressa in entrambe le direzioni (Drug1, Drug2)
% e (Drug2, Drug1) per garantire la simmetria della valutazione.
% ======================================================================

% Interazioni note per classe farmacologica.
% Formato: dangerous_classes(ClasseATC1, ClasseATC2, Severità, Messaggio).
dangerous_classes('n02a', 'n05c', 'high', 'CRITICAL: Respiratory depression risk (Opioids + Sedatives)').
dangerous_classes('b01',  'm01a', 'high', 'CRITICAL: Severe bleeding risk (Anticoagulants + NSAIDs)').
dangerous_classes('n06ab','n06af','high', 'CRITICAL: Serotonin syndrome risk').

% Proprietà fisiopatologiche degli effetti sulla pressione arteriosa.
increases_bp(Drug) :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'r03').
decreases_bp(Drug) :- has_atc_code(Drug, Atc), sub_atom(Atc, 0, 3, _, 'c07').

% 1. Antagonismo fisiopatologico sulla pressione arteriosa.
check_pair_safety(Drug1, Drug2, conflict('high', 'CRITICAL: Physiological Antagonism (Opposite BP effects)')) :-
    increases_bp(Drug1),
    decreases_bp(Drug2).

check_pair_safety(Drug1, Drug2, conflict('high', 'CRITICAL: Physiological Antagonism (Opposite BP effects)')) :-
    decreases_bp(Drug1),
    increases_bp(Drug2).

% 2. DDI da classi farmacologiche pericolose.
check_pair_safety(Drug1, Drug2, conflict(Severity, Msg)) :-
    has_atc_code(Drug1, Atc1),
    has_atc_code(Drug2, Atc2),
    dangerous_classes(C1, C2, Severity, Msg),
    sub_atom(Atc1, 0, _, _, C1),
    sub_atom(Atc2, 0, _, _, C2).

check_pair_safety(Drug1, Drug2, conflict(Severity, Msg)) :-
    has_atc_code(Drug1, Atc1),
    has_atc_code(Drug2, Atc2),
    dangerous_classes(C1, C2, Severity, Msg),
    sub_atom(Atc2, 0, _, _, C1),
    sub_atom(Atc1, 0, _, _, C2).

% 3. Cascata prescrittiva: FANS che induce ipertensione secondaria
%    trattata con ACE-inibitore, generando un ciclo iatrogeno evitabile.
check_pair_safety(Drug1, Drug2, conflict('medium', 'WARNING: Prescribing Cascade detected.')) :-
    has_atc_code(Drug1, Atc1), sub_atom(Atc1, 0, 4, _, 'm01a'),
    has_atc_code(Drug2, Atc2), sub_atom(Atc2, 0, 4, _, 'c09a').

check_pair_safety(Drug1, Drug2, conflict('medium', 'WARNING: Prescribing Cascade detected.')) :-
    has_atc_code(Drug1, Atc1), sub_atom(Atc1, 0, 4, _, 'c09a'),
    has_atc_code(Drug2, Atc2), sub_atom(Atc2, 0, 4, _, 'm01a').

% 4. Duplicazione terapeutica: due farmaci della stessa sottoclasse ATC
%    (prime 4 cifre), che implica stesso meccanismo d'azione.
check_pair_safety(Drug1, Drug2, conflict('medium', 'WARNING: Therapeutic Duplication (Same pharmacological subgroup)')) :-
    Drug1 \= Drug2,
    has_atc_code(Drug1, Atc1),
    has_atc_code(Drug2, Atc2),
    sub_atom(Atc1, 0, 4, _, CoreClass),
    sub_atom(Atc2, 0, 4, _, CoreClass).

% 5. Fallback: nessuna delle condizioni precedenti è verificata.
check_pair_safety(_, _, safe).