from django.test import SimpleTestCase


class HomePageTests(SimpleTestCase):
    def test_homepage_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Hub da análise')

    def test_graphics_page_loads(self):
        response = self.client.get('/graficos/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Amazon Book Rating Dataset')

    def test_sentiment_page_loads(self):
        response = self.client.get('/sentimento/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Análise de Sentimento')
