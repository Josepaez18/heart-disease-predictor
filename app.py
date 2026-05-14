# app.py - Prediccion de Enfermedades Cardiacas
# Modelos: Regresion Logistica y Red Neuronal Artificial (ANN)
# Dataset: Heart Disease UCI
# Ejecutar: python app.py

import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ----- Configuracion -----
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def entrenar_modelos():
    """Entrena ambos modelos desde heart.csv en memoria (sin archivos)."""
    print("  Entrenando modelos desde heart.csv ...")

    df = pd.read_csv(os.path.join(BASE_DIR, 'heart.csv')).dropna()
    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Modelo 1: Regresion Logistica
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    acc_lr = (lr.predict(X_test_sc) == y_test).mean()
    print(f"  Regresion Logistica  Accuracy: {acc_lr*100:.2f}%")

    # Modelo 2: Red Neuronal ANN
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    ann.fit(X_train_sc, y_train)
    acc_ann = (ann.predict(X_test_sc) == y_test).mean()
    print(f"  Red Neuronal (ANN)   Accuracy: {acc_ann*100:.2f}%")

    print("  Modelos entrenados (en memoria)\n")
    return lr, ann, sc, acc_lr, acc_ann


# ----- Cargar modelos al iniciar -----
logreg_model, ann_model, scaler, acc_lr, acc_ann = entrenar_modelos()


# ----- Rutas -----

@app.route('/')
def index():
    """Pagina principal con accuracy de cada modelo."""
    return render_template('index.html',
                           acc_lr=round(acc_lr * 100, 1),
                           acc_ann=round(acc_ann * 100, 1))


@app.route('/predict', methods=['POST'])
def predict():
    """Prediccion individual de un paciente."""
    data = request.get_json()
    modelo = data.get('model', 'lr')
    features = data.get('features', [])

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({'error': 'Caracteristicas invalidas.'}), 400

    if len(features) != 13:
        return jsonify({'error': 'Se requieren exactamente 13 caracteristicas.'}), 400

    X_sc = scaler.transform(X)

    if modelo == 'lr':
        pred = int(logreg_model.predict(X_sc)[0])
        prob = float(logreg_model.predict_proba(X_sc)[0][1])
        model_name = 'Regresion Logistica'
        accuracy = acc_lr
    else:
        pred = int(ann_model.predict(X_sc)[0])
        prob = float(ann_model.predict_proba(X_sc)[0][1])
        model_name = 'Red Neuronal ANN'
        accuracy = acc_ann

    return jsonify({
        'prediction': pred,
        'probability': prob,
        'model': model_name,
        'accuracy': round(accuracy * 100, 1)
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Prediccion por lotes: recibe un archivo CSV subido."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se envio ningun archivo.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacio.'}), 400

    modelo = request.form.get('model', 'lr')

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    except Exception as e:
        return jsonify({'error': f'Error al leer CSV: {str(e)}'}), 400

    if 'target' in df.columns:
        df = df.drop('target', axis=1)

    cols = df.columns.tolist()
    rows = df.values.tolist()

    try:
        X = np.array(rows, dtype=float)
    except (ValueError, TypeError):
        return jsonify({'error': 'El CSV contiene datos no numericos.'}), 400

    X_sc = scaler.transform(X)

    if modelo == 'lr':
        preds = logreg_model.predict(X_sc).tolist()
        probs = logreg_model.predict_proba(X_sc)[:, 1].tolist()
    else:
        preds = ann_model.predict(X_sc).tolist()
        probs = ann_model.predict_proba(X_sc)[:, 1].tolist()

    return jsonify({
        'columns': cols,
        'rows': rows,
        'predictions': preds,
        'probabilities': probs
    })


# ----- Iniciar servidor -----

if __name__ == '__main__':
    print("=" * 50)
    print("  CardioPredict - Enfermedades Cardiacas")
    print("=" * 50)
    print("  http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
