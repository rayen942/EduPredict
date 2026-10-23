"""
Tests unitaires pour EduPredict
Valide les endpoints API et modèles
"""

import json
import unittest
from app3 import CORS_ORIGINS, app


class PredictionAPITest(unittest.TestCase):
    """Test suite pour l'API de prédiction"""

    def setUp(self):
        """Configuration du test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        """Test GET /health"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'ok')
        self.assertIn('tensorflow', data)
        self.assertIn('model', data)
        self.assertIn('kpi_count', data)

    def test_root_endpoint(self):
        """Test GET /"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('endpoints', data)

    def test_info_endpoint(self):
        """Test GET /info"""
        response = self.client.get('/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('endpoints', data)

    def test_cors_allows_configured_dashboard_origin(self):
        """The dashboard origin is allowed by the configured CORS policy."""
        self.assertTrue(CORS_ORIGINS)
        response = self.client.get(
            '/health',
            headers={'Origin': CORS_ORIGINS[0]},
        )
        self.assertEqual(
            response.headers.get('Access-Control-Allow-Origin'),
            CORS_ORIGINS[0],
        )

    def test_cors_rejects_unconfigured_origin(self):
        """Origins outside the configured allowlist receive no CORS permission."""
        origin = 'https://untrusted.example'
        self.assertNotIn(origin, CORS_ORIGINS)
        response = self.client.get('/health', headers={'Origin': origin})
        self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))

    def test_predict_valid_payload(self):
        """Test POST /predict avec données valides (format: lignes par année)"""
        payload = {
            'criteria_values': [
                [16, 14, 13, 6, 12, 55, 29],
                [17, 4, 12, 7, 11, 57, 30],
            ],
            'success_percentages': [66.49, 68.52],
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('meta', data)
        # Vérifier que prédiction est entre 0-100
        self.assertGreaterEqual(data['prediction'], 0)
        self.assertLessEqual(data['prediction'], 100)

    def test_predict_invalid_payload(self):
        """Test POST /predict avec données invalides"""
        payload = {
            'criteria_values': [16, 14],  # Trop court
            'success_percentages': [66.49, 68.52],
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 422)

    def test_predict_missing_fields(self):
        """Test POST /predict sans champs requis"""
        payload = {'criteria_values': [16, 14, 13, 6, 12, 55, 29]}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 422)

    def test_predict_compare_endpoint(self):
        """Test POST /predict/compare"""
        payload = {
            'criteria_values': [
                [16, 14, 13, 6, 12, 55, 29],
                [17, 4, 12, 7, 11, 57, 30],
            ],
            'success_percentages': [66.49, 68.52],
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        response = self.client.post(
            '/predict/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        self.assertIn('linear', data['models'])
        self.assertIn('ridge', data['models'])
        self.assertIn('consensus', data)

    def test_kpi_impact_endpoint(self):
        """Test POST /kpi/impact"""
        payload = {
            'criteria_values': [
                [16, 14, 13, 6, 12, 55, 29],
                [17, 4, 12, 7, 11, 57, 30],
            ],
            'success_percentages': [66.49, 68.52],
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        response = self.client.post(
            '/kpi/impact',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('base_prediction', data)
        self.assertIn('impacts', data)

    def test_kpi_definitions_endpoint(self):
        """Test GET /kpi/definitions"""
        response = self.client.get('/kpi/definitions')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('kpis', data)
        self.assertIn('keys_order', data)
        self.assertEqual(len(data['keys_order']), 8)

    def test_history_endpoint(self):
        """Test GET /history"""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('years', data)
        self.assertIn('reussiteGenerale', data)

    def test_404_unknown_route(self):
        """Test 404 sur route inconnue"""
        response = self.client.get('/unknown/route')
        self.assertEqual(response.status_code, 404)

    def test_method_not_allowed(self):
        """Test 405 sur méthode non autorisée"""
        response = self.client.put('/predict')
        self.assertEqual(response.status_code, 405)


class DataValidationTest(unittest.TestCase):
    """Tests validation des données"""

    def test_valid_payload_accepted(self):
        """Un payload cohérent est validé et converti sans erreur"""
        from app3 import validate_predict_payload

        valid = {
            'criteria_values': [[16, 14, 13, 6, 12, 55, 29],
                                [16.95, 4, 12, 7, 11, 57, 30]],
            'success_percentages': [66.49, 68.52],
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        cv, sp, nyc = validate_predict_payload(valid)
        self.assertEqual(len(cv), 2)
        self.assertEqual(len(nyc), 7)

    def test_out_of_range_success_rejected(self):
        """Un taux de réussite hors de [0, 100] lève ValidationError"""
        from app3 import validate_predict_payload, ValidationError

        invalid = {
            'criteria_values': [[16, 14, 13, 6, 12, 55, 29],
                                [16.95, 4, 12, 7, 11, 57, 30]],
            'success_percentages': [66.49, 142.0],   # 142 % invalide
            'next_year_criteria': [16.95, 15, 14, 8, 12, 69, 28.2]
        }
        with self.assertRaises(ValidationError):
            validate_predict_payload(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
