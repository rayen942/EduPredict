#!/usr/bin/env python
"""
Smoke test — flux complet "Établissements" (Frontend → API → Backend → DB)
==========================================================================
Teste l'API en marche (par défaut http://localhost:5000) :

  1. GET  /health                       → serveur vivant
  2. POST /etablissements  (nouvel id)  → création (HTTP 201) + KPI persistés
  3. GET  /etablissements/<id>          → la ligne existe réellement en base
  4. POST /etablissements  (même id)    → mise à jour (upsert) d'une valeur
  5. GET  /etablissements/<id>          → la mise à jour est bien persistée
  6. DELETE /etablissements/<id>        → nettoyage (aucune donnée résiduelle)
  7. GET  /etablissements/<id>          → 404 confirmé

Lancer (API déjà démarrée) :
  python smoke_test_etablissements.py
  python smoke_test_etablissements.py --url http://localhost:5000

Code de sortie : 0 = tout OK, 1 = au moins un échec.
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

DEFAULT_URL = "http://localhost:5000"
TEST_ID = "TN-SMOKE-TEST"

passed = 0
failed = 0


def _req(method, url, body=None, timeout=15):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="Base URL de l'API")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print("=" * 64)
    print(f"  SMOKE TEST — Établissements  ({base})")
    print("=" * 64)

    payload = {
        "nom_etablissement": "Lycee Smoke Test",
        "identifiant": TEST_ID,
        "region": "Sfax",
        "type_etablissement": "Lycee",
        "adresse": "Rue de Test, Sfax",
        "charge_horaire_prof": 16.95, "prof_quittant": 4,
        "activites_culturelles": 14, "actes_violants": 7,
        "eleves_excellent": 12, "eleve_abondonnant": 57,
        "surcharge_classe": 30, "reussite_generale": 68.52,
        "prediction_kpi": 69.0,
    }

    # 1. health
    print("\n[1] GET /health")
    try:
        st, body = _req("GET", f"{base}/health", timeout=5)
        check("serveur répond 200", st == 200, f"(status={st})")
        check("status == ok", body.get("status") == "ok", f"(body={body})")
    except Exception as e:
        check("serveur joignable", False, f"({e})")
        print("\n>>> API injoignable — démarrez-la : python app3.py")
        return _summary()

    # 2. create
    print("\n[2] POST /etablissements (création, nouvel id)")
    st, body = _req("POST", f"{base}/etablissements", payload)
    check("HTTP 201 Created", st == 201, f"(status={st}, body={body})")
    check("réponse contient un id", isinstance(body.get("id"), int) and body["id"] > 0,
          f"(body={body})")

    # 3. read-back
    print("\n[3] GET /etablissements/<id> (la ligne existe vraiment)")
    st, body = _req("GET", f"{base}/etablissements/{TEST_ID}")
    check("HTTP 200", st == 200, f"(status={st})")
    check("reussite_generale persistée == 68.52",
          body.get("reussite_generale") == 68.52, f"(got={body.get('reussite_generale')})")
    check("region persistée == Sfax", body.get("region") == "Sfax",
          f"(got={body.get('region')})")
    check("prediction_kpi == 69.0", body.get("prediction_kpi") == 69.0,
          f"(got={body.get('prediction_kpi')})")

    # 4. update (upsert)
    print("\n[4] POST /etablissements (mise à jour, même id)")
    payload["reussite_generale"] = 70.0
    payload["region"] = "Tunis"
    st, body = _req("POST", f"{base}/etablissements", payload)
    check("HTTP 201 (upsert)", st == 201, f"(status={st}, body={body})")

    # 5. verify update
    print("\n[5] GET /etablissements/<id> (mise à jour persistée)")
    st, body = _req("GET", f"{base}/etablissements/{TEST_ID}")
    check("reussite_generale == 70.0", body.get("reussite_generale") == 70.0,
          f"(got={body.get('reussite_generale')})")
    check("region mise à jour == Tunis", body.get("region") == "Tunis",
          f"(got={body.get('region')})")

    # 6. delete (cleanup)
    print("\n[6] DELETE /etablissements/<id> (nettoyage)")
    st, body = _req("DELETE", f"{base}/etablissements/{TEST_ID}")
    check("HTTP 200 supprimé", st == 200, f"(status={st}, body={body})")

    # 7. confirm gone
    print("\n[7] GET /etablissements/<id> (404 attendu)")
    st, body = _req("GET", f"{base}/etablissements/{TEST_ID}")
    check("HTTP 404 introuvable", st == 404, f"(status={st})")

    return _summary()


def _summary():
    print("\n" + "=" * 64)
    print(f"  RÉSULTAT : {passed} réussi(s), {failed} échec(s)")
    print("=" * 64)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
