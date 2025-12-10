import requests

def test_diagno():
    files = {'file': open('tests/sample_leaf.jpg', 'rb')}
    response = requests.post("http://localhost:8000/diagno", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "disease" in data
    assert "confidence" in data
    assert "mlflow_uri" in data
    assert "api_uri" in data

# Test prédiction réelle (à activer plus tard)
# def test_predict():
#     files = {'file': open('tests/sample_leaf.jpg', 'rb')}
#     response = requests.post("http://localhost:8000/predict", files=files)
#     assert response.status_code == 200
#     data = response.json()
#     assert "disease" in data
#     assert "confidence" in data
#     assert "mlflow_uri" in data
#     assert "api_uri" in data