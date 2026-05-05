# ❤️ CardioPredict — Predictor de Enfermedad Cardíaca

**Proyecto Final Individual · Entrega: 14 de Mayo 2026**

Aplicación web que predice la presencia de enfermedad cardíaca usando dos modelos de Machine Learning entrenados con el dataset Cleveland del UCI.

---

## 🌐 Demo en vivo
> Sube a GitHub Pages y pon el link aquí.

---

## 🚀 Cómo desplegar en GitHub Pages

### Paso 1 — Crea un repositorio en GitHub
1. Ve a [github.com](https://github.com) → **New repository**
2. Nombre: `heart-disease-predictor` (o el que prefieras)
3. Visibilidad: **Public**
4. Crea el repositorio (sin inicializar)

### Paso 2 — Sube los archivos
Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
git init
git add .
git commit -m "Initial commit - CardioPredict"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/heart-disease-predictor.git
git push -u origin main
```

### Paso 3 — Activa GitHub Pages
1. En tu repositorio → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / `/ (root)`
4. Clic en **Save**
5. En ~1 minuto tu app estará en:
   `https://TU_USUARIO.github.io/heart-disease-predictor/`

---

## 📁 Estructura del proyecto

```
heart-disease-app/
├── index.html      ← Página web principal (todo en uno)
├── models.json     ← Modelos exportados (LR + ANN + Scaler)
├── heart.csv       ← Dataset para predicción por lotes de muestra
└── README.md       ← Este archivo
```

---

## 🧠 Modelos

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|--------|----------|-----------|--------|----|---------|
| Regresión Logística | 88.9% | 85.7% | 92.3% | 88.9% | 96.4% |
| Red Neuronal (MLP 64-32) | 85.2% | 82.1% | 88.5% | 85.2% | 90.7% |

Entrenados con **scikit-learn** en Python, exportados a JSON y ejecutados directamente en el browser (sin backend).

---

## 📊 Dataset

- **Fuente:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Versión:** Cleveland (303 pacientes, 13 características)
- **Target:** 0 = Sin enfermedad, 1 = Con enfermedad cardíaca

### Variables

| Variable | Descripción |
|----------|-------------|
| age | Edad del paciente (años) |
| sex | Sexo (1=M, 0=F) |
| cp | Tipo de dolor torácico (1-4) |
| trestbps | Presión arterial en reposo (mmHg) |
| chol | Colesterol sérico (mg/dl) |
| fbs | Azúcar en ayunas > 120 mg/dl |
| restecg | Resultados ECG en reposo |
| thalach | Frecuencia cardíaca máxima |
| exang | Angina inducida por ejercicio |
| oldpeak | Depresión ST por ejercicio |
| slope | Pendiente del segmento ST pico |
| ca | Número de vasos principales (0-3) |
| thal | Tipo de defecto cardíaco |

---

## ⚠️ Aviso

Este proyecto es académico. Los resultados no deben usarse como diagnóstico médico.
