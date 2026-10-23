"""
db_check.py — Vérification de la connexion PostgreSQL
======================================================
Inspecte le schéma 'edupredict' et affiche les données de la table
'etablissements' si elle existe et contient des données.

Usage :
  python db_check.py

Prérequis :
  pip install sqlalchemy psycopg2-binary
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError, ProgrammingError

# Charge un éventuel fichier .env pour l'usage autonome (python db.py).
# Dépendance optionnelle : sans python-dotenv, on utilise l'environnement du shell.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Les identifiants se lisent depuis l'environnement (voir .env.example).
# Ne jamais utiliser ni committer de mot de passe par défaut.

DB_CONFIG = {
    "drivername": "postgresql+psycopg2",
    "username":   os.environ.get("DB_USER", "postgres"),
    "password":   os.environ.get("DB_PASSWORD", ""),
    "host":       os.environ.get("DB_HOST", "localhost"),
    "port":       int(os.environ.get("DB_PORT", "5432")),
    "database":   os.environ.get("DB_NAME", "prédection CTNE"),  # accents/espaces gérés par URL.create
}

# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    url = URL.create(**DB_CONFIG)
    print(f"Connexion à : {url.render_as_string(hide_password=True)}")

    try:
        engine = create_engine(url, pool_pre_ping=True)
    except Exception as exc:
        print(f"[ERREUR] Impossible de créer le moteur SQLAlchemy : {exc}")
        return

    with engine.connect() as conn:
        # ── Schemas disponibles ───────────────────────────────────────────────
        inspector = inspect(conn)
        schemas = inspector.get_schema_names()
        print(f"\nSchémas disponibles : {schemas}")

        target_schema = "edupredict"
        if target_schema not in schemas:
            print(
                f"\n[AVERTISSEMENT] Le schéma '{target_schema}' n'existe pas.\n"
                f"Schémas trouvés : {schemas}\n"
                "Créez-le avec :  CREATE SCHEMA edupredict;"
            )
            return

        # ── Tables dans le schéma ─────────────────────────────────────────────
        tables = inspector.get_table_names(schema=target_schema)
        print(f"\nTables dans '{target_schema}' : {tables}")

        target_table = "etablissements"
        if target_table not in tables:
            print(
                f"\n[AVERTISSEMENT] La table '{target_schema}.{target_table}' n'existe pas.\n"
                f"Tables disponibles : {tables}"
            )
            return

        # ── Comptage des lignes ────────────────────────────────────────────────
        count_result = conn.execute(
            text(f"SELECT COUNT(*) FROM {target_schema}.{target_table}")
        )
        row_count = count_result.scalar_one()
        print(f"\nNombre de lignes dans '{target_schema}.{target_table}' : {row_count}")

        if row_count == 0:
            print("La table est vide. Ajoutez des données ou vérifiez une autre table.")
            return

        # ── Affichage des premières lignes ────────────────────────────────────
        result = conn.execute(
            text(f"SELECT * FROM {target_schema}.{target_table} LIMIT 20")
        )
        rows = result.fetchall()
        cols = list(result.keys())

        print(f"\nColonnes : {cols}")
        print(f"\n{'─' * 60}")
        for row in rows:
            print(dict(zip(cols, row)))
        print(f"{'─' * 60}")
        if row_count > 20:
            print(f"… {row_count - 20} ligne(s) supplémentaire(s) non affichée(s).")


# ──────────────────────────────────────────────────────────────────────────────
# GESTION DE LA TABLE etablissements_kpi
# ──────────────────────────────────────────────────────────────────────────────

def create_kpi_table(engine) -> None:
    """
    Crée la table edupredict.etablissements_kpi si elle n'existe pas.
    Colonnes : id (serial PK), nom_etablissement (text), identifiant (text unique),
               charge_horaire_prof, prof_quittant, activites_culturelles,
               actes_violants, eleves_excellent, eleve_abondonnant,
               surcharge_classe, reussite_generale (float),
               prediction_kpi (float), date_saisie (timestamp).
    """
    ddl = """
    CREATE SCHEMA IF NOT EXISTS edupredict;

    CREATE TABLE IF NOT EXISTS edupredict.etablissements (
        identifiant         TEXT PRIMARY KEY,
        code                TEXT,
        nom                 TEXT,
        nom_etablissement   TEXT,
        region              TEXT,
        type_etablissement  TEXT,
        adresse             TEXT
    );

    CREATE TABLE IF NOT EXISTS edupredict.etablissements_kpi (
        id                      SERIAL PRIMARY KEY,
        nom_etablissement       TEXT        NOT NULL,
        identifiant             TEXT        NOT NULL UNIQUE
                                            REFERENCES edupredict.etablissements (identifiant)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
        charge_horaire_prof     FLOAT,
        prof_quittant           FLOAT,
        activites_culturelles   FLOAT,
        actes_violants          FLOAT,
        eleves_excellent        FLOAT,
        eleve_abondonnant       FLOAT,
        surcharge_classe        FLOAT,
        reussite_generale       FLOAT,
        prediction_kpi          FLOAT,
        date_saisie             TIMESTAMP   DEFAULT NOW()
    );

    -- Compatibilité : ajoute les colonnes manquantes si la table existait déjà.
    ALTER TABLE edupredict.etablissements ADD COLUMN IF NOT EXISTS region             TEXT;
    ALTER TABLE edupredict.etablissements ADD COLUMN IF NOT EXISTS type_etablissement TEXT;
    ALTER TABLE edupredict.etablissements ADD COLUMN IF NOT EXISTS adresse            TEXT;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("[OK] Table edupredict.etablissements_kpi prête.")


def ensure_parent_etablissement_sql():
    """
    Upsert minimal de la ligne parente dans edupredict.etablissements.

    La table etablissements_kpi porte une contrainte de clé étrangère
    (etablissements_kpi_identifiant_fkey) vers etablissements(identifiant).
    Un KPI ne peut donc être inséré que si l'établissement parent existe déjà.
    On garantit ici sa présence (colonnes NOT NULL : code, nom, identifiant).
    """
    return text("""
        INSERT INTO edupredict.etablissements
            (identifiant, code, nom, nom_etablissement, region, type_etablissement, adresse)
        VALUES
            (:identifiant, :code, :nom, :nom_etablissement, :region, :type_etablissement, :adresse)
        ON CONFLICT (identifiant) DO UPDATE SET
            nom_etablissement  = EXCLUDED.nom_etablissement,
            region             = EXCLUDED.region,
            type_etablissement = EXCLUDED.type_etablissement,
            adresse            = EXCLUDED.adresse
    """)


def insert_etablissement_kpi(engine, data: dict) -> int:
    """
    Insère ou met à jour (upsert) un enregistrement dans etablissements_kpi.
    data doit contenir : nom_etablissement, identifiant, et les clés KPI.

    Garantit d'abord l'existence de l'établissement parent (contrainte FK)
    dans la même transaction, puis upsert les KPI.
    Retourne l'id de la ligne insérée/mise à jour.
    """
    upsert_sql = text("""
        INSERT INTO edupredict.etablissements_kpi
            (nom_etablissement, identifiant,
             charge_horaire_prof, prof_quittant, activites_culturelles,
             actes_violants, eleves_excellent, eleve_abondonnant,
             surcharge_classe, reussite_generale, prediction_kpi, date_saisie)
        VALUES
            (:nom_etablissement, :identifiant,
             :charge_horaire_prof, :prof_quittant, :activites_culturelles,
             :actes_violants, :eleves_excellent, :eleve_abondonnant,
             :surcharge_classe, :reussite_generale, :prediction_kpi, NOW())
        ON CONFLICT (identifiant) DO UPDATE SET
            nom_etablissement     = EXCLUDED.nom_etablissement,
            charge_horaire_prof   = EXCLUDED.charge_horaire_prof,
            prof_quittant         = EXCLUDED.prof_quittant,
            activites_culturelles = EXCLUDED.activites_culturelles,
            actes_violants        = EXCLUDED.actes_violants,
            eleves_excellent      = EXCLUDED.eleves_excellent,
            eleve_abondonnant     = EXCLUDED.eleve_abondonnant,
            surcharge_classe      = EXCLUDED.surcharge_classe,
            reussite_generale     = EXCLUDED.reussite_generale,
            prediction_kpi        = EXCLUDED.prediction_kpi,
            date_saisie           = NOW()
        RETURNING id
    """)
    parent_params = {
        "identifiant":        data["identifiant"],
        "nom_etablissement":  data.get("nom_etablissement"),
        "nom":                data.get("nom_etablissement") or data["identifiant"],
        "code":               "AUTO-" + str(data["identifiant"]),
        "region":             data.get("region"),
        "type_etablissement": data.get("type_etablissement"),
        "adresse":            data.get("adresse"),
    }
    with engine.begin() as conn:
        conn.execute(ensure_parent_etablissement_sql(), parent_params)
        result = conn.execute(upsert_sql, data)
        row = result.fetchone()
        return row[0] if row else -1


def get_all_etablissements_kpi(engine) -> list:
    """Retourne tous les enregistrements de etablissements_kpi (avec infos parent)."""
    sql = text("""
        SELECT k.*, e.region, e.type_etablissement, e.adresse
        FROM edupredict.etablissements_kpi k
        LEFT JOIN edupredict.etablissements e USING (identifiant)
        ORDER BY k.date_saisie DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def get_etablissement_by_identifiant(engine, identifiant: str) -> dict | None:
    """Retourne un établissement par son identifiant unique (avec infos parent)."""
    sql = text("""
        SELECT k.*, e.region, e.type_etablissement, e.adresse
        FROM edupredict.etablissements_kpi k
        LEFT JOIN edupredict.etablissements e USING (identifiant)
        WHERE k.identifiant = :identifiant
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"identifiant": identifiant})
        cols = list(result.keys())
        row = result.fetchone()
        return dict(zip(cols, row)) if row else None


def get_historique_kpi(engine, limit: int = 100) -> list:
    """
    Retourne l'archive des KPI (table edupredict.historique_kpi).

    Cette table est alimentée automatiquement par le trigger trg_archiver_kpi
    à chaque mise à jour d'un établissement : elle conserve l'ancienne valeur
    de reussite_generale et de prediction_kpi avec son horodatage. C'est la
    mémoire historique des saisies, distincte des données live d'etablissements_kpi.

    Retourne [] si la table n'existe pas encore (déploiement sans le script SQL).
    """
    sql = text("""
        SELECT h.identifiant,
               e.nom_etablissement,
               h.reussite_generale,
               h.prediction_kpi,
               h.date_enregistrement
        FROM edupredict.historique_kpi h
        LEFT JOIN edupredict.etablissements e USING (identifiant)
        ORDER BY h.date_enregistrement DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"limit": limit})
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]
    except ProgrammingError:
        # Table historique_kpi absente (script SQL non appliqué) — pas bloquant.
        return []


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except OperationalError as exc:
        print(
            f"\n[ERREUR DE CONNEXION]\n{exc}\n\n"
            "Vérifiez :\n"
            "  1. PostgreSQL est bien démarré\n"
            "  2. Le nom d'utilisateur et le mot de passe sont corrects\n"
            "  3. La base de données existe (CREATE DATABASE ...)\n"
            "  4. L'hôte et le port sont accessibles\n"
        )
    except ProgrammingError as exc:
        print(f"\n[ERREUR SQL]\n{exc}")
    except Exception as exc:
        print(f"\n[ERREUR INATTENDUE]\n{exc}")