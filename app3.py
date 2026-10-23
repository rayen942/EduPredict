"""
EduPredict — Backend Python (Production)
=========================================
API Flask pour la prédiction du taux de réussite scolaire.
Modèle : réseau de neurones TensorFlow avec régularisation,
         features d'interaction, scaler robuste, et fallback Ridge.

Endpoints :
  GET  /health            → état du serveur
  POST /predict           → prédiction principale (réseau de neurones)
  POST /predict/compare   → compare les 3 méthodes (NN + Ridge + Linéaire)
  POST /kpi/impact        → analyse de sensibilité par KPI
  GET  /kpi/definitions   → retourne les définitions KPI
  GET  /history           → retourne les données historiques

Lancer (développement) :
  python app.py

Lancer (production via Waitress) :
  python app.py --prod

Requirements :
  pip install flask flask-cors tensorflow-cpu scikit-learn numpy waitress
"""

import os
import re
import sys
import time
import random
import logging
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Charge un éventuel fichier .env (identifiants DB, clé Anthropic) AVANT que db.py
# ou le bloc Anthropic ne lisent os.environ. Dépendance optionnelle : si
# python-dotenv n'est pas installé, on retombe sur les variables d'environnement
# déjà définies dans le shell.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ─── Logging Configuration (Centralisé) ──────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# Sur Windows, la console (et le fichier) utilisent par défaut cp1252, qui ne
# peut pas encoder les emoji des messages de log (🚀, ✅, ⚠️) ni certains
# caractères (—, accents dans les exceptions). Sans cela, chaque ligne concernée
# déclenche un « UnicodeEncodeError / --- Logging error --- » dans le terminal
# (et la ligne est perdue dans le fichier). On force donc l'UTF-8 partout.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Handler fichier
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Handler console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)-8s: %(message)s')
console_handler.setFormatter(console_formatter)

# Logger racine
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"🚀 EduPredict started — Log file: {LOG_FILE}")

# ─── Typing stubs pour les imports optionnels ──────────────────────────────
tf: Optional[Any] = None

# ─── TensorFlow (lazy import) ─────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential          # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping    # type: ignore
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow available — Neural Network model enabled")
except ImportError as e:
    TF_AVAILABLE = False
    logger.warning(f"⚠️  TensorFlow unavailable — Ridge fallback mode. Error: {e}")

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ─── Import helpers base de données ──────────────────────────────────────────
try:
    from sqlalchemy import create_engine as _sa_create_engine
    from sqlalchemy.engine import URL as _SAURL
    from db import (
        DB_CONFIG,
        create_kpi_table,
        insert_etablissement_kpi,
        get_all_etablissements_kpi,
        get_etablissement_by_identifiant,
        get_historique_kpi,
    )
    _sa_url    = _SAURL.create(**DB_CONFIG)
    db_engine  = _sa_create_engine(_sa_url, pool_pre_ping=True)
    create_kpi_table(db_engine)
    DB_AVAILABLE = True
    logger.info("✅ Connexion PostgreSQL établie — table etablissements_kpi prête.")
except Exception as _db_exc:
    DB_AVAILABLE = False
    db_engine    = None
    logger.warning(f"⚠️  PostgreSQL indisponible — endpoints /etablissements désactivés. Raison: {_db_exc}")


# ─── Anthropic (chatbot ancré sur les données) ────────────────────────────────
try:
    import anthropic  # type: ignore
    _ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if _ANTHROPIC_KEY:
        anthropic_client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        CHAT_AVAILABLE = True
        logger.info("✅ Anthropic configuré — endpoint /chat activé.")
    else:
        anthropic_client = None
        CHAT_AVAILABLE = False
        logger.warning("⚠️  ANTHROPIC_API_KEY absente — endpoint /chat désactivé.")
except Exception as _ai_exc:
    anthropic_client = None
    CHAT_AVAILABLE = False
    logger.warning(f"⚠️  SDK anthropic indisponible — endpoint /chat désactivé. Raison: {_ai_exc}")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

KPI_KEYS = [
    "chargeHoraireProf",
    "profQuittant",
    "activitesCulturelles",
    "actesViolants",
    "elevesExcellent",
    "eleveAbondonnant",
    "surchargeClasse",
    "reussiteGenerale",
]

KPI_DEFINITIONS: Dict[str, Dict] = {
    "chargeHoraireProf":     {"label": "Charge horaire d'enseignant",          "formula": "nb heures / nb profs",                   "unit": "h/prof", "direction": "neutral"},
    "profQuittant":          {"label": "Enseignants quittant l'établissement", "formula": "nb partis / total enseignants",           "unit": "nb",     "direction": "negative"},
    "activitesCulturelles":  {"label": "Activités culturelles et sportives",   "formula": "nb activités / nb matières",              "unit": "nb",     "direction": "positive"},
    "actesViolants":         {"label": "Actes violents",                        "formula": "nb actes / nb personnes",                 "unit": "nb",     "direction": "negative"},
    "elevesExcellent":       {"label": "Élèves inscrits avec score ≥ 12",       "formula": "[élèves ≥16 / total inscrits] × 100",     "unit": "%",      "direction": "positive"},
    "eleveAbondonnant":      {"label": "Élèves abandonnant",                    "formula": "nb retirés / total élèves",               "unit": "nb",     "direction": "negative"},
    "surchargeClasse":       {"label": "Taux de surcharge des classes",         "formula": "(nb élèves / nb classes) × 100",          "unit": "%",      "direction": "negative"},
    "reussiteGenerale":      {"label": "Pourcentage réussite générale",         "formula": "(admis – reçus bac) / (total – bac)",     "unit": "%",      "direction": "positive"},
}

# Données historiques 2017 → 2025 (alignées sur le frontend).
# Les KPI complets sont collectés depuis 2023 ; 2025 est la dernière année observée.
# La prévision (réussite générale) cible 2026 et n'est PAS stockée ici.
HISTORICAL_DATA: Dict[str, Any] = {
    "years":               ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
    "reussiteGenerale":    [None, None, None, None, None, None, 66.49, 68.52, 68.0],
    "chargeHoraireProf":   [None, None, None, None, None, None, 16.0,  17.0,  16.95],
    "profQuittant":        [None, None, None, None, None, None, 14.0,  4.0,   15.0],
    "activitesCulturelles":[None, None, None, None, None, None, 13.0,  12.0,  14.0],
    "actesViolants":       [None, None, None, None, None, None, 6.0,   7.0,   8.0],
    "elevesExcellent":     [None, None, None, None, None, None, 12.0,  11.0,  12.0],
    "eleveAbondonnant":    [None, None, None, None, None, None, 55.0,  57.0,  69.0],
    "surchargeClasse":     [None, None, None, None, None, None, 29.0,  30.0,  28.2],
    "pourcentageSucces":   [47.83, 26.88, 37.01, 32.43, 31.0, 50.39, 42.19, 38.41, None],
}

# ─── Silence TF verbose logs ───────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# Horodatage de démarrage — sert au calcul de l'uptime exposé par /health.
_START_TIME = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int = SEED) -> None:
    """Fixer tous les seeds pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TF_AVAILABLE:
        tf.keras.utils.set_random_seed(seed)  # type: ignore[union-attr]

def disable_tensorflow_runtime() -> None:
    """Désactive TensorFlow si une erreur runtime se produit."""
    global TF_AVAILABLE
    if TF_AVAILABLE:
        TF_AVAILABLE = False
        logger.warning("TensorFlow runtime disabled. Utilisation du fallback Ridge pour les prochaines requêtes.")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_interaction_features(X: np.ndarray) -> np.ndarray:
    """
    Ajoute des produits croisés et des features polynomiales (degree=2).
    Input:  (n_samples, n_features)
    Output: (n_samples, n_features + n_pairs + n_features)
    """
    n = X.shape[1]
    pairs: List[np.ndarray] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(X[:, i] * X[:, j])

    poly2 = X ** 2
    if pairs:
        interaction_matrix = np.column_stack(pairs)
        return np.hstack([X, interaction_matrix, poly2])
    return np.hstack([X, poly2])


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_nn_model(input_dim: int) -> Any:
    """Réseau dense avec BatchNorm, Dropout et régularisation L2."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow n'est pas disponible.")

    model = Sequential([
        Dense(
            128, activation="relu",
            input_shape=(input_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(0.02),  # type: ignore[union-attr]
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(
            64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.02),  # type: ignore[union-attr]
        ),
        BatchNormalization(),
        Dropout(0.2),

        Dense(
            32, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),  # type: ignore[union-attr]
        ),

        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # type: ignore[union-attr]
        loss="huber",
        metrics=["mae"],
    )
    return model


def predict_nn(
    criteria_values: List[List[float]],
    success_percentages: List[float],
    next_year_criteria: List[float],
) -> Tuple[float, Dict]:
    """Prédiction par réseau de neurones avec data augmentation gaussienne."""
    set_seeds()

    X_raw = np.array(criteria_values, dtype=float)
    y_raw = np.array(success_percentages, dtype=float)

    # ── Data augmentation ────────────────────────────────────────────────────
    # n_aug réduit (200 → 40) : suffisant pour ce très petit jeu de données et
    # bien moins de lignes par époque → entraînement nettement plus rapide.
    n_aug = 40
    noise_std = 0.02
    X_aug_list: List[np.ndarray] = []
    y_aug_list: List[float] = []

    for i in range(len(X_raw)):
        for _ in range(n_aug):
            noise = 1.0 + np.random.normal(0.0, noise_std, X_raw.shape[1])
            X_aug_list.append(X_raw[i] * noise)
            y_aug_list.append(
                float(y_raw[i]) * (1.0 + np.random.normal(0.0, noise_std * 0.3))
            )

    X_aug = np.array(X_aug_list, dtype=float)
    y_aug = np.array(y_aug_list, dtype=float)

    # ── Feature engineering ──────────────────────────────────────────────────
    X_fe_aug  = build_interaction_features(X_aug)
    X_fe_real = build_interaction_features(X_raw)
    X_next_fe = build_interaction_features(
        np.array(next_year_criteria, dtype=float).reshape(1, -1)
    )

    X_all = np.vstack([X_fe_aug, X_fe_real])
    y_all = np.concatenate([y_aug, y_raw])

    # ── Normalisation ─────────────────────────────────────────────────────────
    scaler_X = RobustScaler()
    X_train  = scaler_X.fit_transform(X_all)
    X_pred   = scaler_X.transform(X_next_fe)

    scaler_y = RobustScaler()
    y_train_sc = scaler_y.fit_transform(y_all.reshape(-1, 1)).ravel()

    # ── Entraînement ─────────────────────────────────────────────────────────
    model = build_nn_model(X_train.shape[1])
    # Arrêt anticipé plus réactif + moins d'époques + plus gros batch → plus rapide.
    es = EarlyStopping(monitor="loss", patience=8, min_delta=1e-4, restore_best_weights=True)
    history = model.fit(
        X_train, y_train_sc,
        epochs=80,
        batch_size=64,
        verbose=0,
        callbacks=[es],
    )

    # ── Prédiction + descaling ────────────────────────────────────────────────
    y_pred_sc = model.predict(X_pred, verbose=0)
    y_pred    = scaler_y.inverse_transform(y_pred_sc)[0][0]
    y_pred    = float(np.clip(y_pred, 0.0, 100.0))

    meta: Dict = {
        "epochs_trained":   len(history.history["loss"]),
        "final_loss":       round(float(history.history["loss"][-1]), 6),
        "features_count":   X_train.shape[1],
        "augmented_samples": len(X_train),
        "method":           "neural_network",
    }
    return round(y_pred, 2), meta


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE MODEL (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def predict_ridge(
    criteria_values: List[List[float]],
    success_percentages: List[float],
    next_year_criteria: List[float],
    alpha: float = 1.0,
) -> Tuple[float, Dict]:
    """Régression Ridge avec features d'interaction — fallback si TF absent."""
    X_raw = np.array(criteria_values, dtype=float)
    y_raw = np.array(success_percentages, dtype=float)

    X_aug_list = [X_raw]
    y_aug_list = [y_raw]
    for _ in range(150):
        noise = 1.0 + np.random.normal(0.0, 0.02, X_raw.shape)
        X_aug_list.append(X_raw * noise)
        y_aug_list.append(y_raw * (1.0 + np.random.normal(0.0, 0.006, y_raw.shape)))

    X_all = np.vstack(X_aug_list)
    y_all = np.concatenate(y_aug_list)

    X_fe      = build_interaction_features(X_all)
    X_next_fe = build_interaction_features(
        np.array(next_year_criteria, dtype=float).reshape(1, -1)
    )

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])
    pipe.fit(X_fe, y_all)
    y_pred = float(np.clip(pipe.predict(X_next_fe)[0], 0.0, 100.0))

    return round(y_pred, 2), {"method": "ridge", "alpha": alpha}


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR TREND MODEL (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def predict_linear_trend(
    criteria_values: List[List[float]],
    success_percentages: List[float],
    next_year_criteria: List[float],
) -> Tuple[float, Dict]:
    """Extrapolation linéaire pondérée — baseline transparente."""
    y    = np.array(success_percentages, dtype=float)
    X    = np.array(criteria_values,     dtype=float)
    X_nx = np.array(next_year_criteria,  dtype=float)

    n = len(y)
    slope = 0.0
    if n >= 2:
        t     = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, y, 1)[0])
        trend_value = y[-1] + slope
    else:
        trend_value = y[-1]

    kpi_weights = np.array([-0.1, -0.3, 0.15, -0.25, 0.20, -0.30, -0.15, 0.50])
    # Aligner la taille des poids sur le nombre de features
    w = kpi_weights[: X_nx.shape[0]]

    base       = float(y[-1])
    delta      = X_nx - X[-1]
    norm       = np.abs(X[-1]) + 1e-8
    rel_change = delta / norm
    kpi_effect = float(np.dot(w, rel_change)) * base * 0.08

    pred = trend_value * 0.7 + base * 0.3 + kpi_effect
    pred = float(np.clip(pred, 0.0, 100.0))

    return round(pred, 2), {
        "method":      "linear_trend",
        "trend_slope": round(slope, 4),
        "kpi_effect":  round(kpi_effect, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    pass


def validate_predict_payload(data: Dict) -> Tuple[List, List, List]:
    """Valide et convertit le payload JSON entrant."""
    required = ["criteria_values", "success_percentages", "next_year_criteria"]
    for field in required:
        if field not in data:
            raise ValidationError(f"Champ manquant : '{field}'")

    cv  = data["criteria_values"]
    sp  = data["success_percentages"]
    nyc = data["next_year_criteria"]

    if not isinstance(cv, list) or len(cv) < 1:
        raise ValidationError("'criteria_values' doit être une liste avec au moins 1 ligne.")
    if not all(isinstance(row, list) for row in cv):
        raise ValidationError("'criteria_values' doit être une liste de lignes (listes), une par année.")
    if any(len(row) == 0 for row in cv):
        raise ValidationError("Chaque ligne de 'criteria_values' doit être non vide.")
    if not isinstance(sp, list) or len(sp) != len(cv):
        raise ValidationError("'success_percentages' doit avoir la même longueur que 'criteria_values'.")
    if not isinstance(nyc, list) or len(nyc) == 0:
        raise ValidationError("'next_year_criteria' doit être une liste non vide.")

    n_feat = len(cv[0])
    for i, row in enumerate(cv):
        if len(row) != n_feat:
            raise ValidationError(
                f"criteria_values[{i}] : longueur incohérente ({len(row)} vs {n_feat})."
            )
    if len(nyc) != n_feat:
        raise ValidationError(
            f"'next_year_criteria' doit avoir {n_feat} valeurs (comme les années passées)."
        )

    for i, v in enumerate(sp):
        fv = float(v)
        if not (0.0 <= fv <= 100.0):
            raise ValidationError(
                f"success_percentages[{i}] = {v} hors bornes [0, 100]."
            )

    try:
        cv_f  = [[float(x) for x in row] for row in cv]
        sp_f  = [float(x) for x in sp]
        nyc_f = [float(x) for x in nyc]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Valeur non numérique détectée : {exc}") from exc

    return cv_f, sp_f, nyc_f


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
@app.route("/info", methods=["GET"])
def index():
    """Page d'accueil — liste des endpoints."""
    return jsonify({
        "name":    "EduPredict API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /",
            "GET  /info",
            "GET  /health",
            "POST /predict",
            "POST /predict/compare",
            "POST /kpi/impact",
            "GET  /kpi/definitions",
            "GET  /history",
            "GET  /etablissements",
            "GET  /etablissements/<identifiant>",
            "POST /etablissements",
            "DELETE /etablissements/<identifiant>",
            "POST /chat",
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    """Vérification de l'état du serveur."""
    return jsonify({
        "status":              "ok",
        "tensorflow":          TF_AVAILABLE,
        "tensorflow_version":  tf.__version__ if TF_AVAILABLE else None,  # type: ignore[union-attr]
        "model":               "neural_network" if TF_AVAILABLE else "ridge_fallback",
        "kpi_count":           len(KPI_KEYS),
        "database":            DB_AVAILABLE,
        "chat":                CHAT_AVAILABLE,
        "uptime_seconds":      round(time.time() - _START_TIME, 1),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prédiction principale via réseau de neurones (ou Ridge si TF absent).

    Body JSON :
    {
      "criteria_values":     [[kpi1, …], [kpi1, …]],
      "success_percentages": [succ_an1, succ_an2],
      "next_year_criteria":  [kpi1, …]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON invalide ou absent."}), 400

    try:
        cv, sp, nyc = validate_predict_payload(data)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 422

    try:
        if TF_AVAILABLE:
            try:
                pred, meta = predict_nn(cv, sp, nyc)
            except Exception as exc:
                logger.exception("Échec TensorFlow, bascule sur Ridge.")
                disable_tensorflow_runtime()
                pred, meta = predict_ridge(cv, sp, nyc)
                meta["fallback_from"] = "neural_network"
                meta["error"] = str(exc)
        else:
            pred, meta = predict_ridge(cv, sp, nyc)

        return jsonify({
            "prediction": pred,
            "meta":       meta,
            "input_years": len(cv),
            "input_kpis":  len(nyc),
        })
    except Exception as exc:
        logger.exception("Erreur lors de la prédiction")
        return jsonify({"error": f"Erreur interne : {exc}"}), 500


@app.route("/predict/compare", methods=["POST"])
def predict_compare():
    """Compare les 3 méthodes de prédiction — évalue cohérence et incertitude."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON invalide."}), 400

    try:
        cv, sp, nyc = validate_predict_payload(data)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 422

    results: Dict[str, Any] = {}

    # Linear trend
    try:
        lin_pred, lin_meta = predict_linear_trend(cv, sp, nyc)
        results["linear"] = {"prediction": lin_pred, "meta": lin_meta}
    except Exception as exc:
        results["linear"] = {"error": str(exc)}

    # Ridge
    try:
        ridge_pred, ridge_meta = predict_ridge(cv, sp, nyc)
        results["ridge"] = {"prediction": ridge_pred, "meta": ridge_meta}
    except Exception as exc:
        results["ridge"] = {"error": str(exc)}

    # Neural network (si disponible)
    if TF_AVAILABLE:
        try:
            nn_pred, nn_meta = predict_nn(cv, sp, nyc)
            results["neural_network"] = {"prediction": nn_pred, "meta": nn_meta}
        except Exception as exc:
            logger.exception("Échec TensorFlow pour compare; bascule sur Ridge pour ce modèle.")
            disable_tensorflow_runtime()
            fallback_pred, fallback_meta = predict_ridge(cv, sp, nyc)
            results["neural_network"] = {
                "prediction": fallback_pred,
                "meta": fallback_meta,
                "fallback_from": "neural_network",
            }

    valid_preds = [r["prediction"] for r in results.values() if isinstance(r, dict) and "prediction" in r]
    consensus   = round(float(np.mean(valid_preds)),  2) if valid_preds       else None
    std_dev     = round(float(np.std(valid_preds)),   2) if len(valid_preds) > 1 else 0.0

    return jsonify({
        "models":     results,
        "consensus":  consensus,
        "std_dev":    std_dev,
        "confidence": "high" if std_dev < 2 else "medium" if std_dev < 5 else "low",
    })


@app.route("/kpi/definitions", methods=["GET"])
def kpi_definitions():
    """Retourne les métadonnées de tous les KPI."""
    return jsonify({
        "kpis":       KPI_DEFINITIONS,
        "keys_order": KPI_KEYS,
    })


@app.route("/history", methods=["GET"])
def history():
    """Retourne les données historiques complètes."""
    return jsonify(HISTORICAL_DATA)


@app.route("/kpi/impact", methods=["POST"])
def kpi_impact():
    """
    Analyse de sensibilité : impact de +10 % sur chaque KPI.

    Body JSON : même format que /predict
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON absent."}), 400

    try:
        cv, sp, nyc = validate_predict_payload(data)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 422

    # Prédiction de base
    try:
        base_pred, _ = (
            predict_nn(cv, sp, nyc) if TF_AVAILABLE else predict_ridge(cv, sp, nyc)
        )
    except Exception as exc:
        logger.exception("Échec TensorFlow pour KPI impact; bascule sur Ridge.")
        disable_tensorflow_runtime()
        base_pred, _ = predict_ridge(cv, sp, nyc)

    delta_pct = 0.10
    impacts: List[Dict] = []

    for i, key in enumerate(KPI_KEYS[: len(nyc)]):
        nyc_perturbed    = nyc.copy()
        nyc_perturbed[i] = nyc_perturbed[i] * (1.0 + delta_pct)
        try:
            perturbed_pred, _ = (
                predict_nn(cv, sp, nyc_perturbed)
                if TF_AVAILABLE
                else predict_ridge(cv, sp, nyc_perturbed)
            )
        except Exception:
            logger.exception("Échec TensorFlow lors du calcul d'impact; bascule sur Ridge.")
            disable_tensorflow_runtime()
            perturbed_pred, _ = predict_ridge(cv, sp, nyc_perturbed)

        impact: Optional[float] = round(perturbed_pred - base_pred, 3)

        impacts.append({
            "kpi":                        key,
            "label":                      KPI_DEFINITIONS.get(key, {}).get("label", key),
            "direction":                  KPI_DEFINITIONS.get(key, {}).get("direction", "unknown"),
            "impact_per_10pct_increase":  impact,
        })

    impacts.sort(
        key=lambda x: abs(x["impact_per_10pct_increase"] or 0.0),
        reverse=True,
    )

    return jsonify({
        "base_prediction": base_pred,
        "impacts":         impacts,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ETABLISSEMENTS — CRUD
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/etablissements", methods=["GET"])
def list_etablissements():
    """
    Retourne la liste de tous les établissements avec leurs KPI enregistrés.

    Réponse JSON :
    {
      "etablissements": [ { "id", "nom_etablissement", "identifiant",
                             kpi..., "prediction_kpi", "date_saisie" }, … ],
      "total": int
    }
    """
    if not DB_AVAILABLE:
        return jsonify({"error": "Base de données indisponible."}), 503

    try:
        rows = get_all_etablissements_kpi(db_engine)
        # Convertir les datetime en string pour la sérialisation JSON
        for row in rows:
            if row.get("date_saisie"):
                row["date_saisie"] = str(row["date_saisie"])
        return jsonify({"etablissements": rows, "total": len(rows)})
    except Exception as exc:
        logger.exception("Erreur lecture etablissements_kpi")
        return jsonify({"error": f"Erreur interne : {exc}"}), 500


@app.route("/etablissements/<identifiant>", methods=["GET"])
def get_etablissement(identifiant: str):
    """
    Retourne les données d'un établissement par son identifiant unique.
    """
    if not DB_AVAILABLE:
        return jsonify({"error": "Base de données indisponible."}), 503

    try:
        row = get_etablissement_by_identifiant(db_engine, identifiant)
        if not row:
            return jsonify({"error": f"Établissement '{identifiant}' introuvable."}), 404
        if row.get("date_saisie"):
            row["date_saisie"] = str(row["date_saisie"])
        return jsonify(row)
    except Exception as exc:
        logger.exception("Erreur lecture établissement par identifiant")
        return jsonify({"error": f"Erreur interne : {exc}"}), 500


@app.route("/etablissements", methods=["POST"])
def save_etablissement():
    """
    Enregistre ou met à jour les KPI d'un établissement.

    Body JSON requis :
    {
      "nom_etablissement"   : "Lycée …",
      "identifiant"         : "TN-0001",
      "charge_horaire_prof" : 16.95,
      "prof_quittant"       : 4,
      "activites_culturelles": 14,
      "actes_violants"      : 7,
      "eleves_excellent"    : 12,
      "eleve_abondonnant"   : 57,
      "surcharge_classe"    : 30,
      "reussite_generale"   : 68.52,
      "prediction_kpi"      : 69.0   ← optionnel (résultat du modèle)
    }
    """
    if not DB_AVAILABLE:
        return jsonify({"error": "Base de données indisponible."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON invalide ou absent."}), 400

    required = ["nom_etablissement", "identifiant"]
    missing  = [f for f in required if f not in data or not str(data[f]).strip()]
    if missing:
        return jsonify({"error": f"Champs obligatoires manquants : {missing}"}), 422

    nom = str(data["nom_etablissement"]).strip()
    ident = str(data["identifiant"]).strip()
    if len(nom) > 255 or len(ident) > 64:
        return jsonify({"error": "Nom (≤255) ou identifiant (≤64) trop long."}), 422

    # Bornes alignées sur les contraintes CHECK de la base (évite un 500 sur violation).
    # field -> (min, max)
    kpi_bounds = {
        "charge_horaire_prof":   (0.0, None),
        "prof_quittant":         (0.0, None),
        "activites_culturelles": (0.0, None),
        "actes_violants":        (0.0, None),
        "eleves_excellent":      (0.0, 100.0),   # %
        "eleve_abondonnant":     (0.0, None),    # effectif (compteur) → pas de borne à 100
        "surcharge_classe":      (0.0, None),
        "reussite_generale":     (0.0, 100.0),   # %
    }
    record: Dict[str, Any] = {
        "nom_etablissement":    nom,
        "identifiant":          ident,
        "region":               (str(data["region"]).strip() if data.get("region") else None),
        "type_etablissement":   (str(data["type_etablissement"]).strip() if data.get("type_etablissement") else None),
        "adresse":              (str(data["adresse"]).strip() if data.get("adresse") else None),
    }

    def _coerce(field: str, val: Any, lo, hi):
        if val is None or val == "":
            return None
        try:
            fv = float(val)
        except (TypeError, ValueError):
            raise ValidationError(f"Valeur invalide pour '{field}' : {val}")
        if fv != fv or fv in (float("inf"), float("-inf")):
            raise ValidationError(f"Valeur non finie pour '{field}'.")
        if lo is not None and fv < lo:
            raise ValidationError(f"'{field}' = {fv} < {lo} (hors bornes).")
        if hi is not None and fv > hi:
            raise ValidationError(f"'{field}' = {fv} > {hi} (hors bornes).")
        return fv

    try:
        for field, (lo, hi) in kpi_bounds.items():
            record[field] = _coerce(field, data.get(field), lo, hi)
        record["prediction_kpi"] = _coerce("prediction_kpi", data.get("prediction_kpi"), 0.0, 100.0)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 422

    try:
        new_id = insert_etablissement_kpi(db_engine, record)
        return jsonify({
            "message":   "Établissement enregistré avec succès.",
            "id":        new_id,
            "identifiant": record["identifiant"],
        }), 201
    except Exception as exc:
        logger.exception("Erreur insertion etablissements_kpi")
        return jsonify({"error": f"Erreur interne : {exc}"}), 500


@app.route("/etablissements/<identifiant>", methods=["DELETE"])
def delete_etablissement(identifiant: str):
    """
    Supprime un établissement par son identifiant unique.
    """
    if not DB_AVAILABLE:
        return jsonify({"error": "Base de données indisponible."}), 503

    try:
        from sqlalchemy import text as _text
        with db_engine.begin() as conn:
            result = conn.execute(
                _text("DELETE FROM edupredict.etablissements_kpi WHERE identifiant = :id RETURNING id"),
                {"id": identifiant},
            )
            deleted = result.fetchone()
        if not deleted:
            return jsonify({"error": f"Établissement '{identifiant}' introuvable."}), 404
        return jsonify({"message": f"Établissement '{identifiant}' supprimé.", "id": deleted[0]})
    except Exception as exc:
        logger.exception("Erreur suppression établissement")
        return jsonify({"error": f"Erreur interne : {exc}"}), 500


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def handle_404(error):
    return jsonify({
        "error": "Route non trouvée.",
        "available_endpoints": ["/", "/info", "/health", "/predict", "/predict/compare",
                                 "/kpi/definitions", "/history", "/kpi/impact",
                                 "/etablissements", "/etablissements/<identifiant>"],
    }), 404


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT — assistant ancré uniquement sur les données du projet
# ══════════════════════════════════════════════════════════════════════════════

# ── Connaissance de référence du projet (statique, partagée Claude + local) ────
# Décrit ce qu'est EduPredict, son objectif et sa méthodologie. Sert à la fois au
# contexte envoyé à Claude et aux réponses du moteur local (questions « projet »).
PROJECT_OVERVIEW = (
    "EduPredict est une plateforme tunisienne de Business Intelligence scolaire. "
    "Objectif : aider les établissements (écoles, collèges, lycées) à anticiper "
    "leur taux de réussite générale à partir de 8 indicateurs clés (KPI). "
    "La dernière année observée est 2025 ; la plateforme prévoit la réussite "
    "générale de l'année cible 2026 (non encore observée). "
    "Elle combine un tableau de bord analytique, un moteur de prédiction IA et "
    "une base PostgreSQL des établissements."
)

# Méthodologie des 3 modèles de prédiction (endpoints /predict et /predict/compare).
METHODOLOGY_DESC = (
    "Trois méthodes de prédiction sont disponibles et comparées :\n"
    "1. Réseau de neurones (TensorFlow) — modèle principal : couches denses "
    "128→64→32→1, BatchNorm, Dropout, régularisation L2, features d'interaction "
    "et augmentation gaussienne des données. Repli automatique sur Ridge si "
    "TensorFlow est indisponible.\n"
    "2. Régression Ridge — régression linéaire régularisée avec features "
    "d'interaction (repli robuste, sans dépendance lourde).\n"
    "3. Tendance linéaire — extrapolation pondérée année sur année, baseline "
    "transparente et interprétable.\n"
    "L'endpoint /predict/compare renvoie les 3 prédictions, leur consensus "
    "(moyenne) et un niveau de confiance (haute/moyenne/basse) selon l'écart-type. "
    "L'endpoint /kpi/impact mesure l'effet d'une hausse de +10 % de chaque KPI."
)

# Description du schéma de base de données (schéma PostgreSQL « edupredict »).
DB_SCHEMA_DESC = (
    "Base PostgreSQL, schéma « edupredict ». Tables principales :\n"
    "- etablissements : fiche d'identité (identifiant unique, code, nom, "
    "nom_etablissement, region, type_etablissement, adresse).\n"
    "- etablissements_kpi : KPI live de chaque établissement (charge_horaire_prof, "
    "prof_quittant, activites_culturelles, actes_violants, eleves_excellent, "
    "eleve_abondonnant, surcharge_classe, reussite_generale) + prediction_kpi et "
    "date_saisie. Lié à etablissements par identifiant (clé étrangère).\n"
    "- historique_kpi : archive alimentée automatiquement par un trigger à chaque "
    "mise à jour d'un établissement ; conserve l'ancienne reussite_generale et "
    "prediction_kpi avec leur horodatage.\n"
    "- vue v_etablissements_complet : jointure établissement + KPI, triée par "
    "prediction_kpi décroissante."
)

# ── Identité (empreinte) de l'assistant ───────────────────────────────────────
BOT_NAME = "Rayen"
# Signature « empreinte » que le bot appose sur ses messages d'accueil.
BOT_SIGNATURE = f"— {BOT_NAME} 🇹🇳, l'assistant EduPredict"

CHAT_SYSTEM_PROMPT = (
    f"Tu t'appelles {BOT_NAME}, l'assistant tunisien de la plateforme EduPredict "
    "(analyse et prédiction scolaire). "
    "Tu réponds en français, de façon concise, chaleureuse et factuelle. "
    "Tu peux saluer et conclure avec une touche de dialecte tunisien (derja) — "
    "par exemple « 3aslama », « ahla », « behi », « yâaychek », « besslema » — "
    "mais l'explication des données reste toujours en français clair.\n\n"
    "Tu connais l'ensemble du projet : sa description et son objectif, la "
    "méthodologie des 3 modèles de prédiction, les 8 indicateurs (KPI) et leurs "
    "formules, l'historique annuel des indicateurs, le schéma de la base de "
    "données, les établissements enregistrés et l'archive de leurs saisies.\n\n"
    "RÈGLE ABSOLUE : tu ne peux répondre QU'À PARTIR des données et connaissances "
    "fournies ci-dessous (description du projet, méthodologie, définitions des KPI, "
    "données historiques, schéma de base, établissements enregistrés et archive). "
    "Si la réponse ne se trouve pas dans ces données, réponds exactement : "
    "« Je n'ai pas cette information dans les données du projet. » "
    "N'invente JAMAIS de chiffres, de tendances, d'établissements ou de définitions. "
    "Ne réponds pas à des questions générales sans rapport avec EduPredict. "
    "Tu peux faire des calculs simples (moyennes, comparaisons) uniquement sur les chiffres présents.\n\n"
    "===== CONNAISSANCE DU PROJET =====\n"
)


def _build_chat_context(max_rows: int = 200) -> str:
    """
    Assemble TOUTE la connaissance du chatbot à partir des données du projet :
    description, méthodologie des modèles, définitions des KPI, historique annuel
    (avec statistiques calculées), schéma de base, établissements enregistrés et
    archive des saisies. C'est l'unique source de vérité fournie à Claude.
    """
    parts: List[str] = []

    # 1. Présentation et objectif du projet
    parts.append("## Présentation du projet")
    parts.append(PROJECT_OVERVIEW)

    # 2. Méthodologie de prédiction (3 modèles)
    parts.append("\n## Méthodologie de prédiction")
    parts.append(METHODOLOGY_DESC)

    # 3. Définitions des KPI
    parts.append("\n## Définitions des indicateurs (KPI)")
    for key in KPI_KEYS:
        d = KPI_DEFINITIONS.get(key)
        if not d:
            continue
        parts.append(
            f"- {key} — {d['label']} | formule: {d['formula']} | "
            f"unité: {d['unit']} | sens favorable: {d['direction']}"
        )

    # 4. Données historiques + statistiques calculées (min/max/moyenne/tendance)
    parts.append("\n## Données historiques (par année)")
    years = HISTORICAL_DATA.get("years", [])
    parts.append(f"Années couvertes: {', '.join(years)}. Cible de prévision: 2026.")
    for key, series in HISTORICAL_DATA.items():
        if key == "years":
            continue
        pairs = [(y, v) for y, v in zip(years, series) if v is not None]
        if not pairs:
            continue
        détail = ", ".join(f"{y}={v}" for y, v in pairs)
        vals = [float(v) for _, v in pairs]
        tendance = ("hausse" if vals[-1] > vals[0]
                    else "baisse" if vals[-1] < vals[0] else "stable")
        stats = (f" [min={min(vals):.2f}, max={max(vals):.2f}, "
                 f"moyenne={sum(vals) / len(vals):.2f}, tendance={tendance}]")
        parts.append(f"- {key}: {détail}{stats}")

    # 5. Schéma de la base de données
    parts.append("\n## Schéma de la base de données")
    parts.append(DB_SCHEMA_DESC)

    # 6. Établissements enregistrés en base (données live)
    parts.append("\n## Établissements enregistrés en base de données")
    if DB_AVAILABLE and db_engine is not None:
        try:
            rows = get_all_etablissements_kpi(db_engine)
            if not rows:
                parts.append("(Aucun établissement enregistré pour le moment.)")
            else:
                shown = min(len(rows), max_rows)
                parts.append(f"Total: {len(rows)} établissement(s). "
                             f"Détail des {shown} plus récents :")
                for row in rows[:max_rows]:
                    fields = []
                    for k, v in row.items():
                        if k in ("id",) or v is None:
                            continue
                        fields.append(f"{k}={v}")
                    parts.append("- " + ", ".join(fields))
                if len(rows) > max_rows:
                    parts.append(f"… et {len(rows) - max_rows} établissement(s) "
                                 "supplémentaire(s) non détaillé(s).")
        except Exception as exc:
            logger.warning(f"Chat: lecture DB échouée — {exc}")
            parts.append("(Base de données momentanément indisponible.)")
    else:
        parts.append("(Base de données non connectée — aucune donnée d'établissement disponible.)")

    # 7. Archive des saisies (table historique_kpi, alimentée par trigger)
    parts.append("\n## Archive des saisies (historique_kpi)")
    if DB_AVAILABLE and db_engine is not None:
        try:
            archive = get_historique_kpi(db_engine, limit=max_rows)
            if not archive:
                parts.append("(Aucune saisie archivée pour le moment.)")
            else:
                parts.append(f"{len(archive)} enregistrement(s) archivé(s) "
                             "(anciennes valeurs conservées à chaque mise à jour) :")
                for h in archive:
                    nom = h.get("nom_etablissement") or h.get("identifiant")
                    parts.append(
                        f"- {nom} ({h.get('identifiant')}) : "
                        f"reussite_generale={h.get('reussite_generale')}, "
                        f"prediction_kpi={h.get('prediction_kpi')}, "
                        f"le {h.get('date_enregistrement')}"
                    )
        except Exception as exc:
            logger.warning(f"Chat: lecture archive échouée — {exc}")
            parts.append("(Archive momentanément indisponible.)")
    else:
        parts.append("(Base de données non connectée — archive indisponible.)")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# MOTEUR LOCAL (sans LLM) — répond depuis les données du projet, coût nul.
# Utilisé par défaut, et comme fallback si l'API Claude échoue/est absente.
# ──────────────────────────────────────────────────────────────────────────────

NO_INFO_MSG = "Je n'ai pas cette information dans les données du projet."

# Sens favorable → phrase lisible
_DIRECTION_FR = {
    "positive": "une hausse est favorable",
    "negative": "une hausse est défavorable",
    "neutral":  "indicateur neutre",
}

# Alias de recherche → clé KPI (du plus spécifique au plus générique).
_KPI_ALIASES: List[Tuple[str, str]] = [
    ("charge horaire",        "chargeHoraireProf"),
    ("chargehoraireprof",     "chargeHoraireProf"),
    ("enseignants quittant",  "profQuittant"),
    ("prof quittant",         "profQuittant"),
    ("profquittant",          "profQuittant"),
    ("quittant",              "profQuittant"),
    ("activites culturelles", "activitesCulturelles"),
    ("activitesculturelles",  "activitesCulturelles"),
    ("culturelles",           "activitesCulturelles"),
    ("sportives",             "activitesCulturelles"),
    ("actes violents",        "actesViolants"),
    ("actes violants",        "actesViolants"),
    ("actesviolants",         "actesViolants"),
    ("violen",                "actesViolants"),
    ("violan",                "actesViolants"),
    ("eleves excellent",      "elevesExcellent"),
    ("elevesexcellent",       "elevesExcellent"),
    ("excellent",             "elevesExcellent"),
    ("abondonnant",           "eleveAbondonnant"),
    ("abandonnant",           "eleveAbondonnant"),
    ("abandon",               "eleveAbondonnant"),
    ("eleveabondonnant",      "eleveAbondonnant"),
    ("surchargeclasse",       "surchargeClasse"),
    ("surcharge",             "surchargeClasse"),
    ("reussite generale",     "reussiteGenerale"),
    ("reussitegenerale",      "reussiteGenerale"),
    ("taux de reussite",      "reussiteGenerale"),
    ("reussite",              "reussiteGenerale"),
]


def _norm(s: str) -> str:
    """Minuscule + suppression des accents pour une comparaison robuste."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


# ── Dialecte tunisien (derja) ──────────────────────────────────────────────────
# Salutations/remerciements en derja → déclenchent un accueil chaleureux.
_DERJA_GREETINGS = (
    "3aslama", "3asslema", "aslema", "asslema", "ahla", "ahlan", "ahlen",
    "sba7", "sbe7 el khir", "msa el khir", "labes", "labess", "chna7welek",
    "chna7walek", "chnahwelek", "ya3ishek", "yaaychek", "yâaychek", "3aychek",
    "3ayshek", "ya3aychek", "choukran", "barakallah", "besslema", "bisslema",
    "filaman", "fillaman",
)

# Mots derja → équivalents français ajoutés au message pour réutiliser
# toute la logique d'intention existante (définition, combien, comment…).
_DERJA_EXPANSIONS: List[Tuple[str, str]] = [
    ("chnowa",   "c'est quoi signifie"),
    ("chnuwa",   "c'est quoi signifie"),
    ("chniya",   "c'est quoi signifie"),
    ("chneya",   "c'est quoi signifie"),
    ("chnia",    "c'est quoi signifie"),
    ("9adech",   "combien valeur"),
    ("9addech",  "combien valeur"),
    ("9adesh",   "combien valeur"),
    ("kadech",   "combien valeur"),
    ("9addesh",  "combien valeur"),
    ("kifech",   "comment evolution"),
    ("kifeh",    "comment evolution"),
    ("famma",    "combien liste"),
    ("fama",     "combien liste"),
    ("famech",   "combien liste"),
    ("3andek",   "liste affiche"),
    ("a3tini",   "liste affiche"),
    ("a3tine",   "liste affiche"),
    ("madrasa",  "etablissement ecole"),
    ("m3had",    "etablissement lycee"),
    ("ma3had",   "etablissement lycee"),
    ("talaba",   "eleves"),
    ("talmiz",   "eleves"),
    ("najah",    "reussite"),
    ("m3adel",   "moyenne"),
]


def _expand_derja(nmsg: str) -> str:
    """Augmente le message normalisé avec les équivalents français des mots derja."""
    extra = [fr for term, fr in _DERJA_EXPANSIONS if term in nmsg]
    return (nmsg + " " + " ".join(extra)).strip() if extra else nmsg


def _find_kpi(nmsg: str) -> Optional[str]:
    for term, key in _KPI_ALIASES:
        if term in nmsg:
            return key
    return None


def _series_pairs(key: str) -> List[Tuple[str, Any]]:
    """Paires (année, valeur) non nulles pour un indicateur historique."""
    years = HISTORICAL_DATA.get("years", [])
    series = HISTORICAL_DATA.get(key)
    if not series:
        return []
    return [(y, v) for y, v in zip(years, series) if v is not None]


def _live_rows() -> List[Dict[str, Any]]:
    if DB_AVAILABLE and db_engine is not None:
        try:
            return get_all_etablissements_kpi(db_engine)
        except Exception as exc:
            logger.warning(f"Chat local: lecture DB échouée — {exc}")
    return []


def _live_archive(limit: int = 30) -> List[Dict[str, Any]]:
    if DB_AVAILABLE and db_engine is not None:
        try:
            return get_historique_kpi(db_engine, limit=limit)
        except Exception as exc:
            logger.warning(f"Chat local: lecture archive échouée — {exc}")
    return []


def _kpi_full_answer(key: str) -> str:
    d = KPI_DEFINITIONS[key]
    direction = _DIRECTION_FR.get(d["direction"], d["direction"])
    txt = (f"{d['label']} ({key}) : calculé par « {d['formula']} », "
           f"exprimé en {d['unit']} — {direction}.")
    pairs = _series_pairs(key)
    if pairs:
        y, v = pairs[-1]
        txt += f" Dernière valeur connue : {v} ({y})."
    return txt


def local_chat_answer(message: str) -> str:
    """Répond à une question en s'appuyant uniquement sur les données du projet."""
    nmsg = _norm(message)
    if not nmsg:
        return ("Posez-moi une question sur les KPI, l'historique des indicateurs "
                "ou les établissements enregistrés.")

    # Mémorise la présence de derja avant d'enrichir le message,
    # puis traduit les mots derja en intentions françaises connues.
    is_derja = any(term in nmsg for term in _DERJA_GREETINGS) or \
        any(term in nmsg for term, _ in _DERJA_EXPANSIONS)
    nmsg = _expand_derja(nmsg)

    # 1. Salutation en derja (touche tunisienne)
    if any(term in nmsg for term in _DERJA_GREETINGS):
        return (f"3aslama ! Ana {BOT_NAME}, msa3dek mta3 EduPredict 🇹🇳. "
                "Nesta3ref bel KPI, l'évolution mta3 les indicateurs w les "
                "établissements enregistrés. Chnowa t7eb ta3ref ?\n"
                f"{BOT_SIGNATURE}")

    # 1 bis. Salutation classique (français)
    if re.search(r"\b(bonjour|salut|bonsoir|coucou|hello|hey|cc)\b", nmsg):
        return (f"Bonjour ! Je suis {BOT_NAME}, votre assistant EduPredict. Je peux "
                "vous renseigner sur les définitions des KPI, l'évolution historique "
                "des indicateurs et les établissements enregistrés en base. "
                f"Que souhaitez-vous savoir ?\n{BOT_SIGNATURE}")

    # 2. Aide / capacités
    if ("que peux" in nmsg or "tu peux faire" in nmsg or "aide" in nmsg
            or "capac" in nmsg or "tu fais quoi" in nmsg or "comment ca marche" in nmsg):
        return ("Je connais tout le projet EduPredict. Exemples de questions :\n"
                "• « C'est quoi EduPredict ? » / « Quel est l'objectif ? »\n"
                "• « Comment fonctionne la prédiction ? » / « Quels modèles ? »\n"
                "• « Que signifie le KPI surchargeClasse ? »\n"
                "• « Quelle est l'évolution du taux de réussite ? »\n"
                "• « Quelle est la moyenne de reussiteGenerale ? »\n"
                "• « Combien d'établissements sont enregistrés ? »\n"
                "• « Quel est le meilleur / le pire établissement ? »\n"
                "• « Liste les établissements » / « Montre l'archive des saisies »")

    # 2 bis. Présentation / objectif du projet
    if (("c'est quoi" in nmsg or "cest quoi" in nmsg or "qu'est" in nmsg
         or "quest" in nmsg or "presente" in nmsg or "presentation" in nmsg
         or "objectif" in nmsg or "but du projet" in nmsg or "a quoi sert" in nmsg)
            and ("edupredict" in nmsg or "projet" in nmsg or "plateforme" in nmsg
                 or "application" in nmsg or "appli" in nmsg)):
        return PROJECT_OVERVIEW

    # 2 ter. Méthodologie / modèles de prédiction
    if (("modele" in nmsg or "modeles" in nmsg or "methode" in nmsg
         or "methodo" in nmsg or "algorithme" in nmsg or "reseau de neurone" in nmsg
         or "neural" in nmsg or "ridge" in nmsg or "tensorflow" in nmsg
         or "intelligence artificielle" in nmsg or " ia " in f" {nmsg} ")
            or ("comment" in nmsg and ("predi" in nmsg or "prevoi" in nmsg
                                       or "prevision" in nmsg or "calcul" in nmsg))):
        return METHODOLOGY_DESC

    has_etab = "etablissement" in nmsg or "ecole" in nmsg or "lycee" in nmsg

    # 2 quater. Archive des saisies (table historique_kpi)
    if ("archive" in nmsg or "historique des saisie" in nmsg
            or "saisies archiv" in nmsg or "anciennes valeurs" in nmsg):
        archive = _live_archive()
        if not archive:
            return ("L'archive des saisies (historique_kpi) est vide pour le moment. "
                    "Elle se remplit automatiquement à chaque mise à jour d'un établissement.")
        lignes = [
            f"{h.get('nom_etablissement') or h.get('identifiant')} "
            f"({h.get('identifiant')}) : réussite={h.get('reussite_generale')}, "
            f"prédiction={h.get('prediction_kpi')} — le {h.get('date_enregistrement')}"
            for h in archive[:20]
        ]
        suffix = "" if len(archive) <= 20 else f"\n… et {len(archive) - 20} autre(s)."
        return ("Archive des saisies (anciennes valeurs conservées par le trigger) :\n- "
                + "\n- ".join(lignes) + suffix)

    # 3. Nombre d'établissements
    if "combien" in nmsg and has_etab:
        rows = _live_rows()
        if not rows:
            return "Aucun établissement n'est enregistré dans la base pour le moment."
        return f"Il y a {len(rows)} établissement(s) enregistré(s) dans la base."

    # 4. Liste des établissements
    if has_etab and any(w in nmsg for w in ("liste", "lister", "quels", "affiche",
                                            "noms", "lesquels", "tous les")):
        rows = _live_rows()
        if not rows:
            return "Aucun établissement n'est enregistré dans la base pour le moment."
        names = [f"{r.get('nom_etablissement')} ({r.get('identifiant')})" for r in rows[:30]]
        suffix = "" if len(rows) <= 30 else f"\n… et {len(rows) - 30} autre(s)."
        return "Établissements enregistrés :\n- " + "\n- ".join(names) + suffix

    # 4 bis. Classement : meilleur / pire établissement (par prédiction)
    _veut_meilleur = any(t in nmsg for t in ("meilleur", "meilleure", "mieux classe",
                                             "plus haute", "plus haut", "plus eleve",
                                             "plus performant", "top", "premier", "best"))
    _veut_pire = any(t in nmsg for t in ("pire", "moins bon", "moins bonne", "plus faible",
                                         "plus bas", "plus basse", "dernier", "derniere",
                                         "mauvais", "worst", "en difficulte"))
    if (_veut_meilleur or _veut_pire) and (has_etab or "prediction" in nmsg
                                           or "predit" in nmsg or "resultat" in nmsg
                                           or "reussite" in nmsg):
        rows = _live_rows()
        notes = [r for r in rows if r.get("prediction_kpi") is not None]
        if not notes:
            return ("Aucun établissement n'a encore de prédiction enregistrée, "
                    "je ne peux donc pas établir de classement.")
        notes.sort(key=lambda r: float(r["prediction_kpi"]), reverse=True)
        if _veut_pire:
            r = notes[-1]
            rang = "Établissement le plus en difficulté (prédiction la plus faible)"
        else:
            r = notes[0]
            rang = "Meilleur établissement (prédiction la plus élevée)"
        return (f"{rang} : {r.get('nom_etablissement')} ({r.get('identifiant')}) "
                f"avec une prédiction de {float(r['prediction_kpi']):.2f}%"
                + (f", réussite générale {r.get('reussite_generale')}%."
                   if r.get("reussite_generale") is not None else "."))

    # 5. Établissement précis (identifiant ou nom cité dans la question)
    for r in _live_rows():
        ident = _norm(str(r.get("identifiant") or ""))
        nom = _norm(str(r.get("nom_etablissement") or ""))
        if (ident and ident in nmsg) or (len(nom) > 3 and nom in nmsg):
            champs = []
            for key in KPI_KEYS:
                col = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()  # camelCase → snake_case
                val = r.get(col)
                if val is not None:
                    champs.append(f"{KPI_DEFINITIONS[key]['label']} = {val}")
            pred = r.get("prediction_kpi")
            if pred is not None:
                champs.append(f"Prédiction = {pred}")
            corps = "\n- ".join(champs) if champs else "aucun KPI renseigné"
            return (f"{r.get('nom_etablissement')} ({r.get('identifiant')}) :\n- {corps}")

    # 6. Liste des KPI
    if (("liste" in nmsg or "quels" in nmsg or "tous les" in nmsg or "lesquels" in nmsg)
            and ("kpi" in nmsg or "indicateur" in nmsg)):
        lignes = [f"• {KPI_DEFINITIONS[k]['label']} ({k})" for k in KPI_KEYS]
        return "Les 8 indicateurs suivis :\n" + "\n".join(lignes)

    key = _find_kpi(nmsg)

    # 7. Définition d'un KPI
    if any(t in nmsg for t in ("signifie", "definir", "definition", "c'est quoi",
                               "cest quoi", "veut dire", "qu'est", "quest",
                               "represente", "ca veut dire")):
        if key:
            return _kpi_full_answer(key)
        return ("Précisez l'indicateur à définir (ex. surchargeClasse, reussiteGenerale). "
                "Tapez « liste des KPI » pour les voir tous.")

    # 8. Évolution / tendance
    if any(t in nmsg for t in ("evolution", "evolue", "evolu", "tendance",
                               "historique", "progression", "evoluer")):
        k = key or "reussiteGenerale"
        pairs = _series_pairs(k)
        if not pairs:
            return f"Je n'ai pas de série historique pour cet indicateur."
        label = KPI_DEFINITIONS.get(k, {}).get("label", k)
        detail = ", ".join(f"{y} : {v}" for y, v in pairs)
        (fy, fv), (ly, lv) = pairs[0], pairs[-1]
        tendance = "en hausse" if lv > fv else "en baisse" if lv < fv else "stable"
        return (f"Évolution de « {label} » — {detail}. "
                f"Tendance globale {tendance} (de {fv} en {fy} à {lv} en {ly}).")

    # 9. Moyenne
    if "moyenne" in nmsg:
        k = key or "reussiteGenerale"
        pairs = _series_pairs(k)
        if not pairs:
            return "Je n'ai pas de série historique pour calculer cette moyenne."
        vals = [float(v) for _, v in pairs]
        label = KPI_DEFINITIONS.get(k, {}).get("label", k)
        return (f"Moyenne de « {label} » sur {len(vals)} année(s) "
                f"({pairs[0][0]}–{pairs[-1][0]}) : {sum(vals) / len(vals):.2f}.")

    # 10. Valeur précise (année donnée ou dernière connue)
    if key and any(t in nmsg for t in ("valeur", "dernier", "derniere", "actuel",
                                       "combien", "quel est", "quelle est", "2017",
                                       "2018", "2019", "2020", "2021", "2022",
                                       "2023", "2024", "2025")):
        pairs = _series_pairs(key)
        label = KPI_DEFINITIONS[key]["label"]
        unit = KPI_DEFINITIONS[key]["unit"]
        if not pairs:
            return f"Je n'ai pas de valeur historique pour « {label} »."
        m = re.search(r"\b(20(?:1[7-9]|2[0-6]))\b", nmsg)
        if m:
            yr = m.group(1)
            for y, v in pairs:
                if y == yr:
                    return f"« {label} » en {yr} : {v} {unit}."
            return f"Je n'ai pas de valeur de « {label} » pour {yr}."
        y, v = pairs[-1]
        return f"Dernière valeur connue de « {label} » : {v} {unit} ({y})."

    # 11. KPI mentionné seul → définition par défaut
    if key:
        return _kpi_full_answer(key)

    if is_derja:
        return (f"Sma7ni, ma3andich hedhi l'information fi données mta3 le projet. "
                f"As2elni 3al KPI, l'historique wala les établissements. {NO_INFO_MSG}")
    return NO_INFO_MSG


def _real_anthropic_configured() -> bool:
    """Vrai uniquement si une vraie clé Anthropic (api.anthropic.com) est utilisable.

    Le proxy interne de Claude Code (ANTHROPIC_BASE_URL=cc.freemodel.dev) ne sert
    que le client officiel et renvoie 403 : on l'exclut pour ne pas casser /chat.
    """
    if not CHAT_AVAILABLE or anthropic_client is None:
        return False
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip().lower()
    if base and "anthropic.com" not in base:
        return False
    return True


@app.route("/chat", methods=["POST"])
def chat():
    """
    Assistant conversationnel ancré exclusivement sur les données du projet.

    Body JSON :
      { "message": str, "history": [ {"role": "user"|"assistant", "content": str}, ... ] }

    Réponse :
      { "reply": str, "engine": "claude"|"local" }

    Utilise Claude si une vraie clé API est configurée ; sinon (ou en cas d'échec)
    bascule automatiquement sur le moteur local qui répond depuis les données — coût nul.
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Champ 'message' requis."}), 400

    # ── Voie LLM (uniquement si une vraie clé Anthropic est disponible) ──────────
    if _real_anthropic_configured():
        history = data.get("history") or []
        messages: List[Dict[str, str]] = []
        for turn in history[-10:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        system = CHAT_SYSTEM_PROMPT + _build_chat_context()
        try:
            resp = anthropic_client.messages.create(
                model=CHAT_MODEL,
                max_tokens=700,
                system=system,
                messages=messages,
            )
            reply = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            ).strip()
            if reply:
                return jsonify({"reply": reply, "engine": "claude"})
        except Exception as exc:
            logger.warning(f"API Claude indisponible — bascule sur le moteur local. Raison : {exc}")

    # ── Moteur local (par défaut, et fallback) ───────────────────────────────────
    try:
        return jsonify({"reply": local_chat_answer(message), "engine": "local"})
    except Exception as exc:
        logger.exception("Erreur du moteur de chat local")
        return jsonify({"error": f"Erreur de l'assistant : {exc}"}), 500


@app.errorhandler(405)
def handle_405(error):
    return jsonify({"error": "Méthode HTTP non autorisée pour cette route."}), 405


@app.errorhandler(500)
def handle_500(error):
    return jsonify({"error": "Erreur interne du serveur."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _banner() -> None:
    logger.info("═" * 48)
    logger.info("  EduPredict API  —  démarrage")
    logger.info(f"  TensorFlow : {'✓ ' + tf.__version__ if TF_AVAILABLE else '✗ non installé'}")  # type: ignore[union-attr]
    logger.info(f"  Mode       : {'neural_network' if TF_AVAILABLE else 'ridge_fallback'}")
    logger.info(f"  KPI définis: {len(KPI_KEYS)}")
    logger.info(f"  Adresse    : http://{HOST}:{PORT}")
    logger.info("═" * 48)


if __name__ == "__main__":
    _banner()

    prod_mode = "--prod" in sys.argv

    if prod_mode:
        # ── Production : Waitress (multi-thread, stable, pas de rechargement auto)
        try:
            from waitress import serve  # type: ignore
            logger.info("Serveur PRODUCTION démarré via Waitress.")
            serve(app, host=HOST, port=PORT, threads=8)
        except ImportError:
            logger.warning("Waitress non installé — bascule sur Flask dev server.")
            logger.warning("Installe-le avec :  pip install waitress")
            app.run(host=HOST, port=PORT, debug=False)
    else:
        # ── Développement : Flask en mode normal, sans rechargement auto
        logger.info("Serveur DÉVELOPPEMENT démarré (Flask built-in).")
        logger.info("Pour le mode production : python app3.py --prod")
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)