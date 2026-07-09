# 🎓 SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--ML(Q)-green.svg)](https://ml-ops.org/content/crisp-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Descripción

**SIAAP** es un sistema de Machine Learning supervisado que predice el **rendimiento académico** de estudiantes (Bajo, Medio o Alto) a partir de variables de comportamiento, contexto y desempeño previo. El sistema recomienda **intervenciones pedagógicas personalizadas** antes de que el estudiante entre en riesgo académico.

El proyecto implementa:
- 📊 **Regresión Logística** — modelo base interpretable
- 🌲 **Random Forest** — modelo ensemble optimizado  
- 📈 **Regresión Lineal** — análisis de predicción continua del índice de rendimiento
- 🔬 Metodología **CRISP-ML(Q)** — 6 fases completas
- 🖥️ Interfaz interactiva con **Streamlit**
- 🌐 **Landing Page** HTML con imágenes de RA generadas por IA

---

## 📁 Estructura del Proyecto

```
SIAAP_Project/
│
├── 📓 SIAAP_CRISP_ML.ipynb          # Cuaderno principal con todo el pipeline ML
├── 🖥️ app_streamlit.py               # Interfaz interactiva Streamlit
├── 🌐 landing_page.html              # Landing Page del proyecto
├── 📋 README.md                      # Este archivo
│
├── modelo_produccion/                # Artefactos del modelo entrenado
│   ├── modelo_rendimiento_academico.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
└── assets/                           # Imágenes generadas por IA (RA)
    ├── siaap_ar_hero.png
    ├── siaap_ar_ml_brain.png
    ├── siaap_ar_classroom.png
    └── siaap_ar_analytics.png
```

---

## ⚙️ Requisitos del Sistema

### Python
- Python **3.8** o superior

### Librerías requeridas

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.28.0
plotly>=5.17.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/SIAAP.git
cd SIAAP
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit plotly scipy
```

### 4. Ejecutar la aplicación Streamlit

```bash
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### 5. Ejecutar el cuaderno Jupyter

```bash
jupyter notebook SIAAP_CRISP_ML.ipynb
```

---

## 📊 Metodología CRISP-ML(Q)

| Fase | Objetivo | Estado |
|------|----------|--------|
| 1. Business & Data Understanding | Definir el problema y explorar datos | ✅ |
| 2. Data Preparation | Limpieza, ingeniería de variables, partición | ✅ |
| 3. Modeling | Entrenamiento y comparación de modelos | ✅ |
| 4. Evaluation | Métricas, matriz de confusión, interpretabilidad | ✅ |
| 5. Deployment | Empaquetado del modelo, función de recomendación | ✅ |
| 6. Monitoring & Maintenance | Plan de monitoreo y detección de drift | ✅ |

---

## 🎯 Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `horas_estudio_semana` | Numérica | Horas de estudio autónomo semanal |
| `asistencia_pct` | Numérica (%) | Porcentaje de asistencia |
| `nota_previa` | Numérica (0-10) | Promedio del periodo anterior |
| `participacion_foro` | Entera | Intervenciones en foros |
| `horas_plataforma_semana` | Numérica | Uso semanal del LMS |
| `entregas_a_tiempo_pct` | Numérica (%) | % entregas a tiempo |
| `horas_sueno` | Numérica | Promedio horas de sueño |
| `indice_socioeconomico` | Numérica (z-score) | Índice socioeconómico |
| `acceso_internet_estable` | Binaria | Conectividad estable |
| `apoyo_familiar` | Ordinal (1-5) | Percepción de apoyo familiar |
| **`rendimiento`** | **Objetivo** | **Bajo / Medio / Alto** |

---

## 📈 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | 0.533 |
| F1-Score Macro | 0.538 |
| AUC-ROC Macro | 0.711 |

> **Nota:** Los resultados son sobre un dataset sintético con ruido intencional. Con datos reales de un LMS institucional se esperan resultados superiores al umbral F1-macro ≥ 0.75.

---

## 🔬 Variables más Influyentes

1. 📚 `horas_estudio_semana` — 21.2%
2. 📝 `nota_previa` — 11.9%
3. 💡 `indice_compromiso` — 11.1%
4. 💰 `indice_socioeconomico` — 10.9%
5. 🏫 `asistencia_pct` — 10.4%

---

## 🤖 Intervenciones Pedagógicas

| Nivel Predicho | Acción Recomendada |
|---|---|
| 🔴 Bajo | Activar tutorías personalizadas de refuerzo semanal |
| 🟡 Medio | Asignar ruta de aprendizaje adaptativa |
| 🟢 Alto | Ofrecer contenido de profundización y mentorías entre pares |

---

## 👨‍💻 Autor

**Alisandro Made**  
Curso: Machine Learning aplicado a la Educación  
Maestría en Inteligencia Artificial  
Año: 2026

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🔗 Referencias

- [CRISP-ML(Q) Methodology](https://ml-ops.org/content/crisp-ml)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io)
- Studer, S. et al. (2021). *Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology*. Machine Learning and Knowledge Extraction.
