# Ré-export pratique
from .var import load_settings
from .shap_generator import shap_local, shap_global
# (Optionnel) exposer directement les constantes si tu veux importer "prêt à l'emploi"
try:
    _CFG = load_settings()  # lit manet_projet04/settings.yml
    RANDOM_STATE   = _CFG.get("RANDOM_STATE", 42)
    PATH_SIRH      = _CFG.get("PATH_SIRH")
    PATH_EVAL      = _CFG.get("PATH_EVAL")
    PATH_SONDAGE   = _CFG.get("PATH_SONDAGE")
    COL_ID         = _CFG.get("COL_ID", "id_employee")
    TARGET         = _CFG.get("TARGET", "a_quitte_l_entreprise")
    NUM_COLS       = _CFG.get("NUM_COLS", [])
    CAT_COLS       = _CFG.get("CAT_COLS", [])
    SAT_COLS       = _CFG.get("SAT_COLS", [])
    FIRST_VARS     = _CFG.get("FIRST_VARS", [])
    SUBSAMPLE_FRAC = _CFG.get("SUBSAMPLE_FRAC", 0.10)
except Exception:
    # Laisse l'import survivre même si le YAML manque
    pass