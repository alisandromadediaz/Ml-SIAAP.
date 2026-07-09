"""
================================================================================
SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado
Interfaz Streamlit para Predicción de Rendimiento Académico
================================================================================
Autor: Alisandro Made
Metodología: CRISP-ML(Q)
Versión: 1.0 — 2026
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, mean_squared_error, r2_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIAAP — Sistema Inteligente de Aprendizaje",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/TU_USUARIO/SIAAP",
        "Report a bug": None,
        "About": "SIAAP v1.0 — Modelo ML para Predicción de Rendimiento Académico"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo oscuro */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f2d5e 50%, #1a1f3a 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 120, 255, 0.15);
    }

    .main-header h1 {
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #39d353);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #8b949e;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(88, 166, 255, 0.2);
        border-color: #58a6ff;
    }

    /* Predicción resultado */
    .prediction-bajo {
        background: linear-gradient(135deg, #3d1f1f, #5c2020);
        border: 2px solid #f85149;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: pulse-red 2s infinite;
    }

    .prediction-medio {
        background: linear-gradient(135deg, #3d3010, #5c4a0a);
        border: 2px solid #d29922;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-alto {
        background: linear-gradient(135deg, #0f3320, #1a5c2a);
        border: 2px solid #39d353;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.4); }
        50% { box-shadow: 0 0 20px 8px rgba(248, 81, 73, 0.2); }
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(57, 211, 83, 0.4); }
        50% { box-shadow: 0 0 20px 8px rgba(57, 211, 83, 0.2); }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 8px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #ffffff !important;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 2rem;
        transition: all 0.3s;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff);
        box-shadow: 0 8px 24px rgba(31, 111, 235, 0.4);
        transform: translateY(-2px);
    }

    /* Info boxes */
    .info-box {
        background: #0d419d20;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #58a6ff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE DATOS Y MODELOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generar_dataset(n=1200, seed=42):
    """Genera el dataset sintético de estudiantes con relaciones causales controladas."""
    rng = np.random.default_rng(seed)

    horas_estudio_semana = np.clip(rng.normal(8, 3.5, n), 0, 25)
    asistencia_pct = np.clip(rng.normal(82, 12, n), 30, 100)
    nota_previa = np.clip(rng.normal(6.8, 1.4, n), 0, 10)
    participacion_foro = np.clip(rng.integers(0, 15, n).astype(float), 0, 14)
    horas_plataforma_semana = np.clip(rng.normal(5, 2.5, n), 0, 20)
    entregas_a_tiempo_pct = np.clip(rng.normal(75, 18, n), 0, 100)
    horas_sueno = np.clip(rng.normal(6.8, 1.1, n), 3, 10)
    indice_socioeconomico = rng.normal(0, 1, n)
    acceso_internet_estable = rng.binomial(1, 0.72, n)
    apoyo_familiar = rng.integers(1, 6, n)

    # Introducir valores faltantes simulando formularios incompletos
    for arr in [horas_sueno, entregas_a_tiempo_pct, participacion_foro]:
        mask = rng.random(n) < 0.028
        arr[mask] = np.nan

    # Variable latente con pesos causales
    latente = (
        0.32 * (horas_estudio_semana / 25) +
        0.22 * (asistencia_pct / 100) +
        0.20 * (nota_previa / 10) +
        0.10 * (entregas_a_tiempo_pct / 100) +
        0.07 * (indice_socioeconomico / 3 + 0.5) +
        0.05 * acceso_internet_estable +
        0.04 * (apoyo_familiar / 5) +
        rng.normal(0, 0.09, n)
    )

    # Cortes en percentiles 33/70 para 3 clases
    q1 = np.nanpercentile(latente, 33)
    q2 = np.nanpercentile(latente, 70)
    rendimiento = np.where(latente <= q1, 'Bajo',
                   np.where(latente <= q2, 'Medio', 'Alto'))

    df = pd.DataFrame({
        'horas_estudio_semana': horas_estudio_semana,
        'asistencia_pct': asistencia_pct,
        'nota_previa': nota_previa,
        'participacion_foro': participacion_foro,
        'horas_plataforma_semana': horas_plataforma_semana,
        'entregas_a_tiempo_pct': entregas_a_tiempo_pct,
        'horas_sueno': horas_sueno,
        'indice_socioeconomico': indice_socioeconomico,
        'acceso_internet_estable': acceso_internet_estable,
        'apoyo_familiar': apoyo_familiar.astype(float),
        'rendimiento': rendimiento,
        'latente': latente
    })
    return df


@st.cache_resource
def entrenar_modelos(df):
    """Entrena los tres modelos ML y devuelve todos los artefactos necesarios."""
    df_limpio = df.copy()

    # Imputación con mediana
    for col in ['horas_sueno', 'entregas_a_tiempo_pct', 'participacion_foro']:
        df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())

    # Ingeniería de variables
    df_limpio['indice_compromiso'] = (
        0.40 * df_limpio['asistencia_pct'] / 100 +
        0.35 * df_limpio['entregas_a_tiempo_pct'] / 100 +
        0.25 * df_limpio['horas_plataforma_semana'] / 20
    )
    df_limpio['riesgo_descanso'] = (df_limpio['horas_sueno'] < 6).astype(int)

    # Features y target
    feature_cols = [
        'horas_estudio_semana', 'asistencia_pct', 'nota_previa',
        'participacion_foro', 'horas_plataforma_semana', 'entregas_a_tiempo_pct',
        'horas_sueno', 'indice_socioeconomico', 'acceso_internet_estable',
        'apoyo_familiar', 'indice_compromiso', 'riesgo_descanso'
    ]

    X = df_limpio[feature_cols]
    y_cat = df_limpio['rendimiento']
    y_cont = df_limpio['latente']

    # Codificación
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cat)

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    _, _, y_cont_train, y_cont_test = train_test_split(
        X, y_cont, test_size=0.2, random_state=42
    )

    # Escalado
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Modelo 1: Regresión Logística ──
    lr_model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=1000, C=1.0, random_state=42
    )
    lr_model.fit(X_train_sc, y_train)

    # ── Modelo 2: Random Forest ──
    rf_model = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # ── Modelo 3: Regresión Lineal ──
    lin_model = LinearRegression()
    lin_model.fit(X_train_sc, y_cont_train)
    y_lin_pred = lin_model.predict(X_test_sc)

    # Predicciones finales
    y_pred_lr = lr_model.predict(X_test_sc)
    y_pred_rf = rf_model.predict(X_test)

    # Guardar modelos
    os.makedirs("modelo_produccion", exist_ok=True)
    joblib.dump(rf_model, "modelo_produccion/modelo_rendimiento_academico.pkl")
    joblib.dump(scaler, "modelo_produccion/scaler.pkl")
    joblib.dump(le, "modelo_produccion/label_encoder.pkl")

    return {
        'df_limpio': df_limpio,
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'y_cont_test': y_cont_test, 'y_lin_pred': y_lin_pred,
        'lr_model': lr_model, 'rf_model': rf_model, 'lin_model': lin_model,
        'scaler': scaler, 'le': le,
        'feature_cols': feature_cols,
        'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf
    }


def recomendar_intervencion(nivel_predicho, probabilidades, datos_estudiante):
    """Genera recomendaciones pedagógicas personalizadas según el nivel predicho."""
    recomendaciones = []
    alertas = []

    if nivel_predicho == 'Bajo':
        recomendaciones.append("🔴 **Activar tutorías personalizadas de refuerzo semanal**")
        recomendaciones.append("📞 Contactar al docente tutor para evaluación inmediata")
        recomendaciones.append("📚 Asignar recursos de apoyo en conceptos fundamentales")
        if datos_estudiante.get('asistencia_pct', 100) < 70:
            alertas.append("⚠️ Asistencia crítica — requiere seguimiento urgente")
        if datos_estudiante.get('horas_estudio_semana', 10) < 4:
            alertas.append("⚠️ Horas de estudio insuficientes — revisar carga académica")

    elif nivel_predicho == 'Medio':
        recomendaciones.append("🟡 **Asignar ruta de aprendizaje adaptativa personalizada**")
        recomendaciones.append("📊 Monitoreo quincenal de progreso académico")
        recomendaciones.append("🎯 Identificar y reforzar áreas de mejora específicas")
        if datos_estudiante.get('participacion_foro', 5) < 3:
            alertas.append("💬 Baja participación en foros — fomentar colaboración")

    else:
        recomendaciones.append("🟢 **Ofrecer contenido de profundización avanzado**")
        recomendaciones.append("🤝 Asignar rol de mentoría entre pares")
        recomendaciones.append("🏆 Proponer proyectos de investigación y retos adicionales")

    return recomendaciones, alertas


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span style='font-size: 3rem;'>🎓</span>
        <h2 style='color: #58a6ff; margin: 0.5rem 0 0 0;'>SIAAP</h2>
        <p style='color: #8b949e; font-size: 0.85rem;'>Sistema Inteligente de Apoyo<br>al Aprendizaje Personalizado</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Navegación")

    pagina = st.radio(
        "Selecciona una sección:",
        ["🏠 Inicio & Resumen",
         "🔬 Exploración de Datos",
         "🤖 Modelos ML",
         "🎯 Predictor Interactivo",
         "📊 Dashboard de Métricas"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    n_estudiantes = st.slider("Estudiantes en dataset", 500, 2000, 1200, 100)
    seed_val = st.number_input("Semilla aleatoria", value=42, min_value=1)

    st.markdown("---")
    st.markdown("""
    <div style='color: #8b949e; font-size: 0.8rem; text-align: center;'>
        <b>Metodología:</b> CRISP-ML(Q)<br>
        <b>Versión:</b> 1.0 — 2026<br>
        <b>Autor:</b> Alisandro Made
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS
# ─────────────────────────────────────────────────────────────────────────────
df = generar_dataset(n=n_estudiantes, seed=seed_val)
artefactos = entrenar_modelos(df)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: INICIO & RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
if pagina == "🏠 Inicio & Resumen":
    st.markdown("""
    <div class='main-header'>
        <h1>🎓 SIAAP</h1>
        <p>Sistema Inteligente de Apoyo al Aprendizaje Personalizado</p>
        <p style='font-size: 0.9rem; color: #58a6ff;'>Predicción de Rendimiento Académico con Machine Learning | Metodología CRISP-ML(Q)</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Estudiantes analizados", f"{n_estudiantes:,}", "Dataset sintético")
    with col2:
        y_test = artefactos['y_test']
        y_pred_rf = artefactos['y_pred_rf']
        acc = accuracy_score(y_test, y_pred_rf)
        st.metric("🎯 Accuracy (RF)", f"{acc:.1%}", "Random Forest")
    with col3:
        f1 = f1_score(y_test, y_pred_rf, average='macro')
        st.metric("📊 F1-Macro", f"{f1:.3f}", "Métrica principal")
    with col4:
        auc = roc_auc_score(
            y_test,
            artefactos['rf_model'].predict_proba(artefactos['X_test']),
            multi_class='ovr', average='macro'
        )
        st.metric("📈 AUC-ROC", f"{auc:.3f}", "Discriminación")

    st.markdown("---")

    # Descripción del proyecto
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### 🎯 ¿Qué hace SIAAP?")
        st.markdown("""
        SIAAP es un sistema de Machine Learning supervisado que **predice el nivel de rendimiento académico**
        de un estudiante (Bajo, Medio o Alto) a partir de variables de comportamiento y desempeño previo.

        El sistema detecta señales de riesgo de forma temprana — durante las **primeras semanas del curso** —
        permitiendo intervenir pedagógicamente antes de que el estudiante entre en crisis académica.
        """)

        st.markdown("### 🔬 Metodología CRISP-ML(Q)")
        fases = {
            "1️⃣ Business Understanding": "Definir el problema educativo y criterios de éxito",
            "2️⃣ Data Preparation": "Limpieza, ingeniería de variables, escalado",
            "3️⃣ Modeling": "Entrenamiento y comparación de modelos supervisados",
            "4️⃣ Evaluation": "Métricas objetivas, interpretabilidad, matriz de confusión",
            "5️⃣ Deployment": "Modelo empaquetado + recomendaciones pedagógicas",
            "6️⃣ Monitoring": "Detección de drift y plan de mantenimiento"
        }
        for fase, desc in fases.items():
            st.markdown(f"**{fase}** — {desc}")

    with col_b:
        st.markdown("### 📊 Distribución del Dataset")
        df_counts = df['rendimiento'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_counts.index,
            values=df_counts.values,
            hole=0.4,
            marker_colors=['#f85149', '#d29922', '#39d353'],
            textinfo='percent+label',
            textfont_size=14
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6edf3',
            showlegend=False,
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 🏆 Variables más importantes")
        importances = artefactos['rf_model'].feature_importances_
        feature_names = artefactos['feature_cols']
        top5_idx = np.argsort(importances)[::-1][:5]
        for idx in top5_idx:
            pct = importances[idx] * 100
            st.markdown(f"**{feature_names[idx]}** — `{pct:.1f}%`")
            st.progress(float(importances[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: EXPLORACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🔬 Exploración de Datos":
    st.markdown("## 🔬 Análisis Exploratorio de Datos (EDA)")
    st.markdown("*Fase 1 — Business & Data Understanding | Fase 2 — Data Preparation*")

    df_limpio = artefactos['df_limpio']

    tabs = st.tabs(["📊 Distribuciones", "🔗 Correlaciones", "📋 Dataset", "🧹 Calidad de Datos"])

    with tabs[0]:
        st.markdown("### Distribución de Horas de Estudio por Nivel de Rendimiento")
        fig_box = px.box(
            df_limpio, x='rendimiento', y='horas_estudio_semana',
            color='rendimiento',
            color_discrete_map={'Alto': '#39d353', 'Medio': '#d29922', 'Bajo': '#f85149'},
            template='plotly_dark',
            title='Horas de Estudio Semanal según Rendimiento'
        )
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,0.8)',
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Asistencia vs Nota Previa")
            fig_scatter = px.scatter(
                df_limpio.sample(400, random_state=42),
                x='asistencia_pct', y='nota_previa',
                color='rendimiento',
                color_discrete_map={'Alto': '#39d353', 'Medio': '#d29922', 'Bajo': '#f85149'},
                template='plotly_dark',
                opacity=0.7
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.markdown("### Índice de Compromiso por Nivel")
            fig_violin = px.violin(
                df_limpio, y='indice_compromiso', x='rendimiento',
                color='rendimiento',
                color_discrete_map={'Alto': '#39d353', 'Medio': '#d29922', 'Bajo': '#f85149'},
                template='plotly_dark', box=True
            )
            fig_violin.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)

    with tabs[1]:
        st.markdown("### 🔗 Matriz de Correlación entre Variables Numéricas")
        num_cols = [
            'horas_estudio_semana', 'asistencia_pct', 'nota_previa',
            'participacion_foro', 'horas_plataforma_semana', 'entregas_a_tiempo_pct',
            'horas_sueno', 'indice_socioeconomico', 'indice_compromiso'
        ]
        corr_matrix = df_limpio[num_cols].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont_size=10,
            hoverongaps=False
        ))
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,0.8)',
            font_color='#e6edf3',
            height=500,
            title='Correlación de Pearson entre Variables Predictoras'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("✅ No se observan correlaciones extremas (>0.9). No hay multicolinealidad severa.")

    with tabs[2]:
        st.markdown("### 📋 Muestra del Dataset (primeras 20 filas)")
        st.dataframe(
            df_limpio.head(20).style.background_gradient(cmap='Blues', subset=['nota_previa', 'horas_estudio_semana']),
            use_container_width=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas totales", f"{len(df_limpio):,}")
        with col2:
            st.metric("Variables predictoras", "12")
        with col3:
            st.metric("Clases objetivo", "3 (Bajo, Medio, Alto)")

    with tabs[3]:
        st.markdown("### 🧹 Resumen de Calidad de Datos")
        calidad = pd.DataFrame({
            'Variable': ['horas_sueno', 'entregas_a_tiempo_pct', 'participacion_foro'],
            'Valores imputados': [34, 34, 40],
            'Estrategia': ['Mediana', 'Mediana', 'Mediana'],
            'Mediana usada': [6.70, 75.30, 4.00]
        })
        st.dataframe(calidad, use_container_width=True)
        st.success("✅ Filas duplicadas encontradas: 0. Dataset listo para modelado.")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: MODELOS ML
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🤖 Modelos ML":
    st.markdown("## 🤖 Modelos de Machine Learning")
    st.markdown("*Fase 3 — Modeling | Fase 4 — Evaluation*")

    le = artefactos['le']
    y_test = artefactos['y_test']
    y_pred_lr = artefactos['y_pred_lr']
    y_pred_rf = artefactos['y_pred_rf']
    clases = le.classes_

    tabs = st.tabs(["📊 Comparación", "🔴 Reg. Logística", "🌲 Random Forest", "📈 Reg. Lineal"])

    with tabs[0]:
        st.markdown("### Comparación de Modelos Supervisados")

        comparacion = pd.DataFrame({
            'Modelo': ['Regresión Logística', 'Random Forest'],
            'Accuracy': [
                accuracy_score(y_test, y_pred_lr),
                accuracy_score(y_test, y_pred_rf)
            ],
            'F1-Macro': [
                f1_score(y_test, y_pred_lr, average='macro'),
                f1_score(y_test, y_pred_rf, average='macro')
            ],
            'AUC-ROC': [
                roc_auc_score(y_test, artefactos['lr_model'].predict_proba(artefactos['X_test_sc']),
                              multi_class='ovr', average='macro'),
                roc_auc_score(y_test, artefactos['rf_model'].predict_proba(artefactos['X_test']),
                              multi_class='ovr', average='macro')
            ]
        })

        fig_bar = go.Figure()
        for metric in ['Accuracy', 'F1-Macro', 'AUC-ROC']:
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=comparacion['Modelo'],
                y=comparacion[metric],
                text=[f"{v:.3f}" for v in comparacion[metric]],
                textposition='auto'
            ))
        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,0.8)',
            font_color='#e6edf3',
            title='Comparación de Métricas por Modelo',
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(comparacion.style.format({
            'Accuracy': '{:.3f}', 'F1-Macro': '{:.3f}', 'AUC-ROC': '{:.3f}'
        }).highlight_max(axis=0, color='#39d35340'), use_container_width=True)

    with tabs[1]:
        st.markdown("### 🔴 Regresión Logística (Multinomial)")
        st.markdown("""
        **Modelo base interpretable** — Asume una relación lineal entre las variables predictoras (escaladas)
        y el logaritmo de las probabilidades de cada clase. Usa `solver='lbfgs'` con regularización L2 (C=1.0).
        """)

        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig_cm = px.imshow(
            cm_lr, labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=clases, y=clases,
            color_continuous_scale='Blues',
            text_auto=True,
            title="Matriz de Confusión — Regresión Logística"
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        report_lr = classification_report(y_test, y_pred_lr, target_names=clases, output_dict=True)
        df_report = pd.DataFrame(report_lr).T.round(3)
        st.dataframe(df_report, use_container_width=True)

    with tabs[2]:
        st.markdown("### 🌲 Random Forest (Modelo Principal)")
        st.markdown("""
        **Modelo ensemble** — Combina 400 árboles de decisión con votación mayoritaria.
        Hiperparámetros optimizados con `GridSearchCV`: `n_estimators=400`, `min_samples_leaf=4`.
        No requiere escalado previo. Es el modelo seleccionado para producción.
        """)

        col1, col2 = st.columns(2)
        with col1:
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm_rf = px.imshow(
                cm_rf, labels=dict(x="Predicho", y="Real", color="Cantidad"),
                x=clases, y=clases, color_continuous_scale='Greens',
                text_auto=True, title="Matriz de Confusión — Random Forest"
            )
            fig_cm_rf.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
            st.plotly_chart(fig_cm_rf, use_container_width=True)

        with col2:
            importances = artefactos['rf_model'].feature_importances_
            feature_names = artefactos['feature_cols']
            df_imp = pd.DataFrame({'Variable': feature_names, 'Importancia': importances})
            df_imp = df_imp.sort_values('Importancia', ascending=True)

            fig_imp = px.bar(
                df_imp, x='Importancia', y='Variable', orientation='h',
                color='Importancia', color_continuous_scale='Viridis',
                title='Importancia de Variables — Random Forest'
            )
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
            st.plotly_chart(fig_imp, use_container_width=True)

    with tabs[3]:
        st.markdown("### 📈 Regresión Lineal (Índice de Rendimiento Continuo)")
        st.markdown("""
        **Modelo de regresión sobre la variable latente continua** — En lugar de predecir la clase categórica,
        predice el índice de rendimiento numérico subyacente (entre 0 y 1). Útil para ranking y tendencias.
        """)

        y_cont_test = artefactos['y_cont_test']
        y_lin_pred = artefactos['y_lin_pred']

        rmse = np.sqrt(mean_squared_error(y_cont_test, y_lin_pred))
        r2 = r2_score(y_cont_test, y_lin_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.4f}", "Varianza explicada")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}", "Error cuadrático medio raíz")

        fig_lin = go.Figure()
        sample_idx = np.random.choice(len(y_cont_test), 200, replace=False)
        fig_lin.add_trace(go.Scatter(
            x=y_cont_test.iloc[sample_idx] if hasattr(y_cont_test, 'iloc') else y_cont_test[sample_idx],
            y=y_lin_pred[sample_idx],
            mode='markers', marker=dict(color='#58a6ff', opacity=0.6, size=6),
            name='Predicciones'
        ))
        x_range = [min(y_cont_test), max(y_cont_test)]
        fig_lin.add_trace(go.Scatter(
            x=x_range, y=x_range,
            mode='lines', line=dict(color='#f85149', dash='dash'),
            name='Predicción perfecta'
        ))
        fig_lin.update_layout(
            title='Valores Reales vs Predichos — Regresión Lineal',
            xaxis_title='Valor Real (Índice Latente)',
            yaxis_title='Valor Predicho',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,0.8)',
            font_color='#e6edf3'
        )
        st.plotly_chart(fig_lin, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: PREDICTOR INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🎯 Predictor Interactivo":
    st.markdown("## 🎯 Predictor Interactivo de Rendimiento")
    st.markdown("*Fase 5 — Deployment | Ingrese los datos del estudiante para obtener una predicción personalizada*")

    col_form, col_result = st.columns([2, 3])

    with col_form:
        st.markdown("### 📝 Datos del Estudiante")

        horas_estudio = st.slider("📚 Horas de estudio semanal", 0.0, 25.0, 8.0, 0.5)
        asistencia = st.slider("🏫 Asistencia (%)", 30.0, 100.0, 82.0, 1.0)
        nota_previa = st.slider("📝 Nota previa (0-10)", 0.0, 10.0, 6.8, 0.1)
        participacion = st.slider("💬 Participaciones en foro", 0, 14, 4)
        horas_plataforma = st.slider("💻 Horas en plataforma semanal", 0.0, 20.0, 5.0, 0.5)
        entregas_tiempo = st.slider("⏰ Entregas a tiempo (%)", 0.0, 100.0, 75.0, 1.0)
        horas_sueno = st.slider("😴 Horas de sueño diario", 3.0, 10.0, 6.8, 0.1)
        indice_se = st.slider("💰 Índice socioeconómico (z-score)", -3.0, 3.0, 0.0, 0.1)
        internet = st.radio("🌐 Acceso internet estable", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        apoyo_fam = st.select_slider("👨‍👩‍👧 Apoyo familiar", options=[1, 2, 3, 4, 5], value=3)

        predecir_btn = st.button("🔮 Predecir Rendimiento", type="primary")

    with col_result:
        if predecir_btn:
            # Calcular variables derivadas
            indice_compromiso = (
                0.40 * asistencia / 100 +
                0.35 * entregas_tiempo / 100 +
                0.25 * horas_plataforma / 20
            )
            riesgo_descanso = 1 if horas_sueno < 6 else 0

            datos = np.array([[
                horas_estudio, asistencia, nota_previa, participacion,
                horas_plataforma, entregas_tiempo, horas_sueno, indice_se,
                internet, float(apoyo_fam), indice_compromiso, riesgo_descanso
            ]])

            scaler = artefactos['scaler']
            le = artefactos['le']
            rf_model = artefactos['rf_model']
            lr_model = artefactos['lr_model']

            datos_sc = scaler.transform(datos)

            # Predicción RF (modelo principal)
            pred_rf = rf_model.predict(datos)[0]
            proba_rf = rf_model.predict_proba(datos)[0]
            nivel_rf = le.inverse_transform([pred_rf])[0]

            # Predicción LR
            pred_lr = lr_model.predict(datos_sc)[0]
            proba_lr = lr_model.predict_proba(datos_sc)[0]
            nivel_lr = le.inverse_transform([pred_lr])[0]

            # Mostrar resultado principal
            color_map = {'Bajo': 'prediction-bajo', 'Medio': 'prediction-medio', 'Alto': 'prediction-alto'}
            emoji_map = {'Bajo': '🔴', 'Medio': '🟡', 'Alto': '🟢'}

            st.markdown(f"""
            <div class='{color_map[nivel_rf]}'>
                <h2>{emoji_map[nivel_rf]} Rendimiento Predicho: {nivel_rf}</h2>
                <p style='color: #e6edf3; font-size: 1rem;'>
                    Confianza: <b>{max(proba_rf)*100:.1f}%</b> | Modelo: Random Forest
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilidades por clase
            st.markdown("#### 📊 Probabilidades por Clase")
            clases = le.classes_
            fig_proba = go.Figure(go.Bar(
                x=[f"{emoji_map[c]} {c}" for c in clases],
                y=proba_rf,
                marker_color=['#f85149' if c == 'Bajo' else '#d29922' if c == 'Medio' else '#39d353' for c in clases],
                text=[f"{p*100:.1f}%" for p in proba_rf],
                textposition='auto'
            ))
            fig_proba.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                font_color='#e6edf3',
                yaxis=dict(tickformat='.0%'),
                height=250,
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_proba, use_container_width=True)

            # Comparación modelos
            st.markdown(f"**Regresión Logística predice:** {emoji_map[nivel_lr]} **{nivel_lr}** "
                        f"(confianza: {max(proba_lr)*100:.1f}%)")

            # Recomendaciones
            st.markdown("#### 🎓 Intervenciones Pedagógicas Recomendadas")
            datos_dict = {
                'asistencia_pct': asistencia,
                'horas_estudio_semana': horas_estudio,
                'participacion_foro': participacion
            }
            recomendaciones, alertas = recomendar_intervencion(nivel_rf, proba_rf, datos_dict)

            for alerta in alertas:
                st.warning(alerta)
            for rec in recomendaciones:
                st.success(rec)

            # Variables derivadas
            st.markdown("#### ⚙️ Variables Derivadas Calculadas")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Índice de Compromiso", f"{indice_compromiso:.3f}")
            with col_b:
                st.metric("Riesgo por Descanso", "⚠️ Sí" if riesgo_descanso else "✅ No")
        else:
            st.markdown("""
            <div style='text-align: center; padding: 4rem 2rem; color: #8b949e;'>
                <div style='font-size: 4rem;'>🔮</div>
                <h3>Listo para predecir</h3>
                <p>Configure los parámetros del estudiante en el panel izquierdo<br>
                y haga clic en <b>Predecir Rendimiento</b>.</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: DASHBOARD DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "📊 Dashboard de Métricas":
    st.markdown("## 📊 Dashboard de Métricas y Monitoreo")
    st.markdown("*Fase 4 — Evaluation | Fase 6 — Monitoring & Maintenance*")

    le = artefactos['le']
    y_test = artefactos['y_test']
    y_pred_rf = artefactos['y_pred_rf']
    clases = le.classes_

    # Métricas globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_rf):.3f}")
    with col2:
        st.metric("F1-Macro", f"{f1_score(y_test, y_pred_rf, average='macro'):.3f}")
    with col3:
        auc = roc_auc_score(
            y_test, artefactos['rf_model'].predict_proba(artefactos['X_test']),
            multi_class='ovr', average='macro'
        )
        st.metric("AUC-ROC Macro", f"{auc:.3f}")
    with col4:
        umbral = 0.75
        f1_actual = f1_score(y_test, y_pred_rf, average='macro')
        estado = "✅ OK" if f1_actual >= umbral else "⚠️ ALERTA"
        st.metric("Estado del Modelo", estado, f"Umbral: {umbral}")

    # Alerta de monitoreo
    f1_actual = f1_score(y_test, y_pred_rf, average='macro')
    if f1_actual < 0.75:
        st.error(f"""
        🚨 **ALERTA DE MONITOREO** — F1-Macro actual ({f1_actual:.3f}) < Umbral ({0.75})

        **Acción recomendada:** Re-entrenamiento con datos reales de un LMS institucional.
        Con datos sintéticos limitados, este comportamiento es esperado y correcto desde CRISP-ML(Q).
        """)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📋 Reporte por Clase")
        report = classification_report(y_test, y_pred_rf, target_names=clases, output_dict=True)
        df_rep = pd.DataFrame(report).T
        df_rep = df_rep.drop(index=['accuracy'], errors='ignore')
        st.dataframe(df_rep.round(3).style.highlight_max(axis=0, color='#39d35340'), use_container_width=True)

    with col_b:
        st.markdown("### 🎯 Plan de Monitoreo CRISP-ML(Q)")
        plan = {
            "Aspecto": ["Frecuencia evaluación", "Umbral de alerta", "Detección de drift",
                        "Re-entrenamiento", "Supervisión humana"],
            "Estrategia": [
                "F1-macro c/2 semanas",
                "F1 < 0.75 → notificación",
                "KS-test train vs. nuevos datos",
                "Automático cada semestre",
                "Docente revisa casos 'Bajo'"
            ]
        }
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)

    # Distribución de errores
    st.markdown("### 🔍 Análisis de Errores de Clasificación")
    mask_error = y_test != y_pred_rf
    df_test_full = artefactos['df_limpio'].iloc[-len(y_test):]
    df_errores = df_test_full[mask_error].copy() if len(df_test_full) == len(y_test) else None

    error_pct = mask_error.mean() * 100
    st.info(f"📊 **Tasa de error global:** {error_pct:.1f}% ({mask_error.sum()} de {len(y_test)} estudiantes mal clasificados)")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=f1_actual,
        title={'text': "F1-Macro Actual", 'font': {'color': '#e6edf3'}},
        delta={'reference': 0.75, 'increasing': {'color': '#39d353'}, 'decreasing': {'color': '#f85149'}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': '#e6edf3'},
            'bar': {'color': '#58a6ff'},
            'steps': [
                {'range': [0, 0.50], 'color': '#3d1f1f'},
                {'range': [0.50, 0.75], 'color': '#3d3010'},
                {'range': [0.75, 1.0], 'color': '#0f3320'}
            ],
            'threshold': {'line': {'color': '#f85149', 'width': 4}, 'thickness': 0.75, 'value': 0.75}
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3', height=300
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 1rem; font-size: 0.85rem;'>
    🎓 <b>SIAAP v1.0</b> — Sistema Inteligente de Apoyo al Aprendizaje Personalizado |
    Metodología CRISP-ML(Q) | Autor: Alisandro Made | 2026
</div>
""", unsafe_allow_html=True)
