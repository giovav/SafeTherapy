# File: src/kb/utils.py
import os
import re

def to_prolog_atom(text: str) -> str:
    """
    Converte una stringa di testo in un atomo Prolog valido.
    
    Gli atomi Prolog sono sensibili alle maiuscole/minuscole e convenzionalmente
    iniziano con una lettera minuscola. Questa funzione formatta i nomi 
    (es. dei farmaci o delle patologie) convertendoli in minuscolo, sostituendo 
    gli spazi con underscore e rimuovendo i caratteri non consentiti.
    
    Args:
        text (str): La stringa originale da convertire (es. "Aspirin 100mg").
        
    Returns:
        str: L'atomo Prolog formattato (es. "aspirin_100mg"), oppure "unknown" 
             se l'input fornito non è valido.
    """
    if not text or not isinstance(text, str):
        return "unknown"
    
    # Converte in minuscolo e rimuove spazi iniziali/finali
    text = text.strip().lower()
    
    # Sostituisce spazi e trattini con underscore
    text = re.sub(r'[\s\-]+', '_', text)
    
    # Rimuove tutti i caratteri che non sono alfanumerici o underscore
    text = re.sub(r'[^a-z0-9_]', '', text)
    
    if not text:
        return "unknown"
        
    # Assicura che inizi con una lettera per evitare conflitti sintattici in Prolog
    if text[0].isdigit() or text[0] == '_':
        text = 'a_' + text
        
    return text


class ProjectConfig:
    """
    Contenitore centralizzato per la configurazione dei percorsi di sistema.

    Calcola automaticamente la directory radice del progetto a partire dalla
    posizione del file corrente e definisce i percorsi canonici per i dati
    grezzi e i modelli serializzati.

    Attributes:
        base_dir (str): Percorso assoluto della root del progetto.
        data_dir (str): Directory contenente i dataset CSV.
        data_path (str): Percorso completo al dataset FAERS grezzo.
        model_dir (str): Directory di destinazione per i modelli `.pkl`.
        model_path (str): Percorso completo al modello Random Forest serializzato.
    """

    def __init__(self):
        """Inizializza i percorsi di sistema calcolando la root del progetto."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.data_dir = os.path.join(self.base_dir, "data")
        self.data_path = os.path.join(self.data_dir, "faers_50k_examples.csv")

        self.model_dir = os.path.join(self.base_dir, "src", "ml", "models")
        self.model_path = os.path.join(self.model_dir, "rf_faers_model.pkl")

    def check_paths(self) -> tuple[bool, str]:
        """
        Verifica l'esistenza del dataset nel percorso configurato.

        Returns:
            tuple[bool, str]: Una coppia (esito, messaggio) dove esito è True
                se il file esiste, False altrimenti, e messaggio descrive
                l'eventuale errore.
        """
        if not os.path.exists(self.data_path):
            return False, f"Dataset non trovato in: {self.data_path}"
        return True, "OK"


class TextUtils:
    """
    Raccolta di metodi statici per la normalizzazione delle stringhe testuali.

    Utilizzata nella pipeline ETL per garantire che i valori categoriali
    (nomi di farmaci, sesso, patologie) siano uniformi prima dell'encoding
    e del training del modello.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalizza una stringa rimuovendo spazi superflui e convertendola
        in maiuscolo, per garantire match coerenti durante l'encoding.

        Args:
            text (str): La stringa da normalizzare. Accetta anche valori non
                stringa, che vengono trattati come assenti.

        Returns:
            str: La stringa normalizzata in formato UPPERCASE, oppure "NONE"
                se l'input è nullo o non è una stringa.

        Examples:
            >>> TextUtils.clean_text("  aspirin ")
            'ASPIRIN'
            >>> TextUtils.clean_text(None)
            'NONE'
        """
        if not isinstance(text, str):
            return "NONE"
        return text.strip().upper()