# -*- coding: utf-8 -*-
"""
SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado
App de Streamlit que expone el modelo de Machine Learning supervisado
(clasificación multiclase) desarrollado en el notebook CRISP-ML(Q).

Ejecutar localmente:
    streamlit run app_streamlit.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SIAAP · Apoyo al Aprendizaje Personalizado",
    page_icon="🎓",
    layout="wide",
)

INK = "#1B2A44"
ACCENT = "#3E5C8A"
BAJO = "#B6412F"
MEDIO = "#C98A22"
ALTO = "#2E6E4E"
COLOR_MAP = {"Bajo": BAJO, "Medio": MEDIO, "Alto": ALTO}

st.markdown(
    """
    <style>
    .main {background-color: #F1F2ED;}
    .stMetric {background: white; padding: 14px; border-radius: 10px; border: 1px solid #C9CCC0;}
    h1, h2, h3 {color: #1B2A44;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 1) GENERACIÓN DEL DATASET SINTÉTICO (cacheada como dato, no como recurso)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Generando dataset sintético (CRISP-ML · Fase 1)...")
def generar_dataset_estudiantes(n=1200, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    horas_estudio_semana = np.clip(rng.normal(8, 3.5, n), 0, 25)
    asistencia_pct = np.clip(rng.normal(82, 12, n), 30, 100)
    nota_previa = np.clip(rng.normal(6.8, 1.4, n), 0, 10)
    participacion_foro = np.clip(rng.poisson(4, n), 0, 20)
    horas_plataforma_semana = np.clip(rng.normal(5, 2.2, n), 0, 15)
    entregas_a_tiempo_pct = np.clip(rng.normal(75, 18, n), 0, 100)
    horas_sueno = np.clip(rng.normal(6.7, 1.1, n), 3, 10)
    indice_socioeconomico = np.clip(rng.normal(0, 1, n), -2.5, 2.5)
    acceso_internet_estable = rng.binomial(1, 0.78, n)
    apoyo_familiar = rng.integers(1, 6, n)

    latente = (
        0.32 * (horas_estudio_semana / 25) +
        0.22 * (asistencia_pct / 100) +
        0.20 * (nota_previa / 10) +
        0.08 * (participacion_foro / 20) +
        0.07 * (horas_plataforma_semana / 15) +
        0.10 * (entregas_a_tiempo_pct / 100) +
        0.03 * ((horas_sueno - 3) / 7) +
        0.05 * (indice_socioeconomico / 2.5) +
        0.03 * acceso_internet_estable +
        0.04 * (apoyo_familiar / 5) +
        rng.normal(0, 0.09, n)
    )

    q1, q2 = np.quantile(latente, [0.33, 0.70])
    rendimiento = np.where(latente <= q1, "Bajo",
                    np.where(latente <= q2, "Medio", "Alto"))

    df = pd.DataFrame({
        "horas_estudio_semana": horas_estudio_semana.round(1),
        "asistencia_pct": asistencia_pct.round(1),
        "nota_previa": nota_previa.round(2),
        "participacion_foro": participacion_foro.astype(int),
        "horas_plataforma_semana": horas_plataforma_semana.round(1),
        "entregas_a_tiempo_pct": entregas_a_tiempo_pct.round(1),
        "horas_sueno": horas_sueno.round(1),
        "indice_socioeconomico": indice_socioeconomico.round(2),
        "acceso_internet_estable": acceso_internet_estable,
        "apoyo_familiar": apoyo_familiar,
        "rendimiento": rendimiento,
    })

    df["indice_compromiso"] = (
        0.4 * (df["asistencia_pct"] / 100) +
        0.3 * (df["entregas_a_tiempo_pct"] / 100) +
        0.3 * (df["horas_plataforma_semana"] / df["horas_plataforma_semana"].max())
    ).round(3)
    df["riesgo_descanso"] = (df["horas_sueno"] < 6).astype(int)

    return df


FEATURE_COLS = [
    "horas_estudio_semana", "asistencia_pct", "nota_previa", "participacion_foro",
    "horas_plataforma_semana", "entregas_a_tiempo_pct", "horas_sueno",
    "indice_socioeconomico", "acceso_internet_estable", "apoyo_familiar",
    "indice_compromiso", "riesgo_descanso",
]


# ----------------------------------------------------------------------
# 2) ENTRENAMIENTO DE MODELOS (cacheado como recurso: guarda objetos no serializables)
#    IMPORTANTE: no se usa 'multi_class' en LogisticRegression — ese parámetro
#    fue removido en versiones recientes de scikit-learn (>=1.7). Con
#    solver='lbfgs' y más de 2 clases, sklearn ya resuelve multinomial
#    automáticamente, así que omitirlo es compatible con cualquier versión.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando modelos (CRISP-ML · Fase 3)...")
def entrenar_modelos(df: pd.DataFrame):
    le = LabelEncoder()
    y = le.fit_transform(df["rendimiento"])
    X = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)

    modelos = {
        "Regresión Logística": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, random_state=RANDOM_STATE
        ),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados_cv = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X_train_esc, y_train, cv=skf, scoring="f1_macro")
        resultados_cv[nombre] = {"media": scores.mean(), "std": scores.std()}

    mejor_nombre = max(resultados_cv, key=lambda k: resultados_cv[k]["media"])
    modelo_final = modelos[mejor_nombre]
    modelo_final.fit(X_train_esc, y_train)

    y_pred = modelo_final.predict(X_test_esc)
    metricas = {
        "modelo_ganador": mejor_nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "matriz_confusion": confusion_matrix(y_test, y_pred),
        "clases": list(le.classes_),
        "resultados_cv": resultados_cv,
    }

    importancias = None
    if hasattr(modelo_final, "feature_importances_"):
        importancias = pd.Series(modelo_final.feature_importances_, index=FEATURE_COLS)
    elif hasattr(modelo_final, "coef_"):
        importancias = pd.Series(np.abs(modelo_final.coef_).mean(axis=0), index=FEATURE_COLS)

    artefactos = {
        "modelo": modelo_final,
        "scaler": scaler,
        "label_encoder": le,
        "metricas": metricas,
        "importancias": importancias,
    }
    return artefactos


def recomendar_intervencion(nivel_predicho, probabilidades, datos_estudiante):
    recomendaciones = []
    if nivel_predicho == "Bajo":
        recomendaciones.append("🔴 Activar tutorías personalizadas de refuerzo semanal.")
        if datos_estudiante["asistencia_pct"] < 75:
            recomendaciones.append("Contactar al estudiante: asistencia por debajo del umbral crítico.")
        if datos_estudiante["horas_sueno"] < 6:
            recomendaciones.append("Sugerir pausa de bienestar: patrón de descanso insuficiente.")
    elif nivel_predicho == "Medio":
        recomendaciones.append("🟡 Asignar ruta de aprendizaje adaptativa con retos progresivos.")
        recomendaciones.append("Enviar recordatorios automáticos de entregas pendientes.")
    else:
        recomendaciones.append("🟢 Ofrecer contenido de profundización y mentorías entre pares.")

    return {
        "nivel_predicho": nivel_predicho,
        "confianza": {k: float(v) for k, v in probabilidades.items()},
        "recomendaciones": recomendaciones,
    }


def predecir_estudiante(artefactos, datos_dict):
    modelo = artefactos["modelo"]
    scaler = artefactos["scaler"]
    le = artefactos["label_encoder"]

    entrada = pd.DataFrame([datos_dict])[FEATURE_COLS]
    entrada_esc = scaler.transform(entrada)
    pred_cod = modelo.predict(entrada_esc)[0]
    proba = modelo.predict_proba(entrada_esc)[0]
    nivel = le.inverse_transform([pred_cod])[0]
    probabilidades = dict(zip(le.classes_, proba))
    return recomendar_intervencion(nivel, probabilidades, datos_dict)


# ----------------------------------------------------------------------
# PIPELINE: generar datos -> entrenar -> artefactos
# ----------------------------------------------------------------------
df = generar_dataset_estudiantes()
artefactos = entrenar_modelos(df)
metricas = artefactos["metricas"]

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🎓 SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado")
st.caption(
    "Modelo de Machine Learning supervisado (clasificación multiclase) desarrollado "
    "con metodología **CRISP-ML(Q)**. Datos sintéticos con fines demostrativos."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Modelo ganador", metricas["modelo_ganador"])
col_b.metric("F1-score macro (test)", f"{metricas['f1_macro']:.3f}")
col_c.metric("Accuracy (test)", f"{metricas['accuracy']:.3f}")
col_d.metric("Clases", " · ".join(metricas["clases"]))

st.divider()

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_pred, tab_modelo = st.tabs(["🔮 Predicción", "📊 El modelo"])

with tab_pred:
    st.subheader("Simula el perfil de un estudiante")

    col1, col2 = st.columns(2)
    with col1:
        horas_estudio_semana = st.slider("Horas de estudio / semana", 0.0, 25.0, 8.0, 0.5)
        asistencia_pct = st.slider("Asistencia (%)", 30.0, 100.0, 80.0, 1.0)
        nota_previa = st.slider("Nota previa (0–10)", 0.0, 10.0, 6.8, 0.1)
        participacion_foro = st.slider("Participación en foros", 0, 20, 4, 1)
        horas_plataforma_semana = st.slider("Uso semanal de la plataforma (h)", 0.0, 15.0, 5.0, 0.5)
        entregas_a_tiempo_pct = st.slider("Entregas a tiempo (%)", 0.0, 100.0, 75.0, 1.0)

    with col2:
        horas_sueno = st.slider("Horas de sueño diarias", 3.0, 10.0, 6.7, 0.1)
        indice_socioeconomico = st.slider("Índice socioeconómico (z-score)", -2.5, 2.5, 0.0, 0.1)
        acceso_internet_estable = st.selectbox("Acceso a internet estable", ["Sí", "No"], index=0)
        apoyo_familiar = st.slider("Apoyo familiar percibido (1–5)", 1, 5, 3, 1)

    if st.button("🔮 Predecir rendimiento y generar recomendación", type="primary"):
        datos_estudiante = {
            "horas_estudio_semana": horas_estudio_semana,
            "asistencia_pct": asistencia_pct,
            "nota_previa": nota_previa,
            "participacion_foro": participacion_foro,
            "horas_plataforma_semana": horas_plataforma_semana,
            "entregas_a_tiempo_pct": entregas_a_tiempo_pct,
            "horas_sueno": horas_sueno,
            "indice_socioeconomico": indice_socioeconomico,
            "acceso_internet_estable": 1 if acceso_internet_estable == "Sí" else 0,
            "apoyo_familiar": apoyo_familiar,
        }
        datos_estudiante["indice_compromiso"] = round(
            0.4 * (datos_estudiante["asistencia_pct"] / 100)
            + 0.3 * (datos_estudiante["entregas_a_tiempo_pct"] / 100)
            + 0.3 * (datos_estudiante["horas_plataforma_semana"] / 15),
            3,
        )
        datos_estudiante["riesgo_descanso"] = int(datos_estudiante["horas_sueno"] < 6)

        resultado = predecir_estudiante(artefactos, datos_estudiante)
        color = COLOR_MAP[resultado["nivel_predicho"]]

        st.markdown(
            f"### Predicción: <span style='color:{color}'>{resultado['nivel_predicho']}</span>",
            unsafe_allow_html=True,
        )
        for r in resultado["recomendaciones"]:
            st.info(r)

        fig = go.Figure(go.Bar(
            x=list(resultado["confianza"].values()),
            y=list(resultado["confianza"].keys()),
            orientation="h",
            marker_color=[COLOR_MAP[c] for c in resultado["confianza"].keys()],
        ))
        fig.update_layout(
            title="Confianza del modelo por clase",
            xaxis_title="Probabilidad",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_modelo:
    st.subheader("Comparación de modelos (validación cruzada, F1-macro)")
    cv_df = pd.DataFrame([
        {"Modelo": k, "F1-macro (media)": v["media"], "Desviación": v["std"]}
        for k, v in metricas["resultados_cv"].items()
    ])
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.subheader("Matriz de confusión (conjunto de prueba)")
    cm = metricas["matriz_confusion"]
    fig_cm = px.imshow(
        cm, text_auto=True, x=metricas["clases"], y=metricas["clases"],
        color_continuous_scale="Blues", labels=dict(x="Predicción", y="Real", color="Casos"),
    )
    fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    if artefactos["importancias"] is not None:
        st.subheader("Importancia de variables")
        imp = artefactos["importancias"].sort_values(ascending=True)
        fig_imp = px.bar(imp, orientation="h", labels={"value": "Importancia relativa", "index": ""})
        fig_imp.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.caption(
        f"Umbral de calidad definido para producción: F1-macro ≥ 0.75 · "
        f"F1-macro actual: {metricas['f1_macro']:.3f} · "
        f"Estado: {'✅ OK' if metricas['f1_macro'] >= 0.75 else '⚠️ ALERTA — re-entrenamiento recomendado'}"
    )

st.divider()
st.caption(
    "Proyecto académico de Machine Learning supervisado · Metodología CRISP-ML(Q) · "
    "Notebook completo disponible en `/notebook` del repositorio."
)
