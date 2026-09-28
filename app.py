"""Laboratorio web de clustering de vinos (Streamlit).

Permite recorrer el trabajo por secciones y experimentar con los datos y los parámetros:
fuente de datos (CSV de la cátedra, CSV propio o UCI), edición de valores, variables usadas,
escalado, outliers, k, algoritmo e hiperparámetros. Todos los textos se generan a partir de
los resultados, por lo que el informe descargable se recrea con cada cambio.

Uso local:  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import clustering as cl  # noqa: E402
from src import eda  # noqa: E402
from src import interpretation as it  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.data import COLUMNAS, DIR_RAW, SEMILLA, leer_vinos  # noqa: E402

REPO = "https://github.com/fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino"
NBVIEWER = ("https://nbviewer.org/github/fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino"
            "/blob/main/notebooks/clustering_vinos.ipynb")
CSV_CATEDRA = DIR_RAW / "wine-clustering.csv"
ESCALADORES = ["StandardScaler (z-score)", "MinMaxScaler [0, 1]", "RobustScaler (mediana/IQR)", "Sin escalar"]

st.set_page_config(page_title="Clustering de vinos", page_icon="🍷", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1350px;}
.cabecera {background: linear-gradient(90deg, #5a1e2b 0%, #7b2d3b 55%, #a8495c 100%);
           color: #fff; padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.cabecera h1 {color: #fff; margin: 0; font-size: 2rem;}
.cabecera p {color: #f3dde2; margin: .3rem 0 0 0;}
div[data-testid="stMetric"] {background: #fbf5f6; border: 1px solid #ecd9dd; border-radius: 12px;
                             padding: .7rem 1rem;}
.ficha {border-left: 6px solid var(--c); background: #fbf8f8; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;}
.ficha h4 {margin-top: 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=300)
def _png(firma: str, nombre: str, _fabrica) -> bytes:
    """Renderiza una figura a PNG. Se cachea por (configuración, nombre): solo se redibuja si algo cambió."""
    fig = _fabrica()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close("all")
    return buffer.getvalue()


def mostrar(nombre: str, fabrica) -> None:
    """Muestra la figura que produce ``fabrica`` (usando la caché de la configuración actual)."""
    st.image(_png(FIRMA, nombre, fabrica), width="stretch")


def tabla_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte índice y columnas a texto (tablas de contingencia con fila/columna "Total")."""
    t = df.copy()
    t.index = t.index.map(str)
    t.columns = t.columns.map(str)
    t.index.name, t.columns.name = df.index.name, df.columns.name
    return t


def fmt(x: float, dec: int = 3) -> str:
    """Número con coma decimal."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def crear_escalador(nombre: str):
    """Instancia el escalador elegido (``None`` = sin escalar)."""
    return None if nombre == "Sin escalar" else pp.construir_escaladores()[nombre]


@st.cache_data(show_spinner=False)
def cargar(fuente: str, contenido: bytes | None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Carga los datos según la fuente elegida en la barra lateral."""
    if fuente == "uci":
        return leer_vinos(None)
    if fuente == "catedra":
        return leer_vinos(CSV_CATEDRA)
    return leer_vinos(io.BytesIO(contenido))


@st.cache_data(show_spinner="Calculando clustering…")
def analizar(df: pd.DataFrame, variables: tuple, escalador_nombre: str, winsorizar: bool, algoritmo: str,
             init: str, n_init: int, max_iter: int, linkage: str, metrica: str, k_manual: int | None,
             semilla: int) -> dict:
    """Corre todo el pipeline con la configuración elegida y devuelve los resultados."""
    df_modelo = pp.recortar_outliers_iqr(df) if winsorizar else df
    escalador = crear_escalador(escalador_nombre)
    datos = df_modelo[list(variables)]
    if escalador is None:
        df_X = datos.copy()
    else:
        df_X, escalador = pp.escalar(datos, escalador)
    X = df_X.to_numpy()

    # Elección de k (siempre con K-Means, como en el trabajo)
    tabla_k = cl.evaluar_rango_k(X, range(1, 11), semilla)
    k_codo = cl.detectar_codo(tabla_k["inercia (WCSS)"])
    Z = cl.calcular_linkage(X, "ward")
    saltos = cl.saltos_dendrograma(Z)
    resumen_k = cl.resumen_eleccion_k(tabla_k, k_codo, saltos)
    k_consenso = int(resumen_k["k sugerido"].mode()[0])
    k = k_manual or k_consenso

    # K-Means de referencia (con los hiperparámetros elegidos) y modelo final
    kmeans = cl.crear_kmeans(k, semilla, n_init=n_init, init=init, max_iter=max_iter).fit(X)
    if algoritmo == "K-Means":
        etiquetas_crudas = kmeans.labels_
    else:
        etiquetas_crudas = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metrica).fit_predict(X)
    etiquetas, _ = it.ordenar_clusters(etiquetas_crudas, df)
    et_km_ord, _ = it.ordenar_clusters(kmeans.labels_, df)

    # Estabilidad (solo tiene sentido en K-Means, que es estocástico)
    estabilidad = None
    if algoritmo == "K-Means":
        estabilidad = pd.DataFrame([
            cl.estabilidad_semillas(X, k, n_init=n_init, init=init),
            cl.estabilidad_semillas(X, k, n_init=1, init="random"),
        ]).drop_duplicates("configuración").set_index("configuración")

    # Contrastes
    X_std = StandardScaler().fit_transform(datos)
    sin_esc, dominancia = cl.comparar_sin_escalar(datos, X_std, et_km_ord, k, semilla)
    df_wins_esc, _ = pp.escalar(pp.recortar_outliers_iqr(datos))
    outliers = cl.impacto_outliers(df_wins_esc.to_numpy(), et_km_ord, k, semilla)
    jerarquicos, et_jer = cl.comparar_jerarquicos(X, k, et_km_ord)
    et_ward, _ = it.ordenar_clusters(et_jer["Jerárquico ward (euclidean)"], df)

    # Interpretación (siempre sobre las 13 variables originales)
    df_z = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    perfiles = it.perfiles_zscore(df_z, etiquetas)
    centroides = df.groupby(etiquetas).mean().rename_axis("cluster")
    if k == 3:
        fichas = it.fichas_tipos(centroides, perfiles, df.mean(), it.identificar_tipos(perfiles))
    else:
        fichas = it.fichas_genericas(centroides, perfiles, df.mean())
    pca, proy, loadings = it.ajustar_pca(df_X)
    centros_X = df_X.groupby(etiquetas).mean().to_numpy()

    return {
        "X": X, "df_X": df_X, "tabla_k": tabla_k, "k_codo": k_codo, "Z": Z, "saltos": saltos,
        "resumen_k": resumen_k, "k_consenso": k_consenso, "k": k, "etiquetas": etiquetas,
        "silueta": silhouette_score(X, etiquetas), "tamanos": cl.tamanos_clusters(etiquetas),
        "estabilidad": estabilidad, "sin_esc": sin_esc, "dominancia": dominancia, "outliers": outliers,
        "jerarquicos": jerarquicos, "contingencia_ward": cl.tabla_contingencia(et_km_ord, et_ward, "K-Means",
                                                                                "Jerárquico Ward"),
        "ari_km": adjusted_rand_score(et_km_ord, etiquetas), "perfiles": perfiles, "centroides": centroides,
        "fichas": fichas, "nombres": {c: f["nombre"] for c, f in fichas.items()},
        "ranking": it.ranking_discriminantes(df_z, etiquetas), "pca": pca, "proy": proy,
        "loadings": loadings, "centros_X": centros_X,
    }


def generar_informe(cfg: dict, r: dict, post: dict | None, df: pd.DataFrame) -> str:
    """Informe en Markdown recreado con la configuración y los resultados actuales."""
    tk, k = r["tabla_k"], r["k"]
    lineas = [
        "# Clustering de vinos: informe generado",
        f"*Generado el {date.today():%d/%m/%Y} desde la app web del trabajo práctico (Aprendizaje Artificial, CAECE).*\n",
        "## Configuración utilizada\n",
        "| Decisión | Elección |", "|---|---|",
        f"| Datos | {cfg['fuente']} ({df.shape[0]} vinos{', con ediciones manuales' if cfg['editado'] else ''}) |",
        f"| Variables para clusterizar | {len(cfg['variables'])} de 13: {', '.join(cfg['variables'])} |",
        f"| Escalado | {cfg['escalador']} |",
        f"| Outliers | {'Winsorizados (1,5·IQR)' if cfg['winsorizar'] else 'Conservados'} |",
        f"| k | {k} ({'consenso automático' if cfg['k_manual'] is None else 'elegido manualmente'}; "
        f"consenso de los criterios = {r['k_consenso']}) |",
        f"| Algoritmo | {cfg['algoritmo_desc']} |",
        f"| Semilla | {cfg['semilla']} |\n",
        "## Elección de k\n", r["resumen_k"].to_markdown(), "",
        tk.round(3).to_markdown(), "",
        "## Resultado\n",
        f"- Silueta del modelo final: **{fmt(r['silueta'])}**",
        f"- Tamaños: {', '.join(f'C{c} = {n}' for c, n in r['tamanos']['vinos'].items())}",
        f"- Acuerdo con K-Means (ARI): {fmt(r['ari_km'])}",
    ]
    if r["estabilidad"] is not None:
        e = r["estabilidad"].iloc[0]
        lineas.append(f"- Estabilidad entre 10 semillas: ARI medio {fmt(e['ARI medio'])}, mínimo {fmt(e['ARI mínimo'])}")
    lineas += ["", "## Tipos de vino\n",
               it.fichas_a_markdown(r["fichas"], r["tamanos"]["vinos"]),
               "## Centroides (unidades originales)\n", r["centroides"].round(2).to_markdown(), ""]
    if post:
        lineas += ["## Validación post-hoc (no interviene en el aprendizaje)\n", post["contingencia"].to_markdown(), "",
                   f"ARI = {fmt(post['ARI'])} · NMI = {fmt(post['NMI'])} · Pureza = {fmt(100 * post['Pureza'], 1)} %"]
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------
# Barra lateral: datos y parámetros
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🍷 Laboratorio de clustering")
    st.link_button("📓 Ver notebook original (nbviewer)", NBVIEWER, width="stretch")
    st.link_button("💻 Repositorio en GitHub", REPO, width="stretch")

    st.markdown("### 1 · Datos")
    opcion = st.radio("Fuente", ["CSV de la cátedra (wine-clustering.csv)", "Subir mi propio CSV",
                                 "Dataset UCI (scikit-learn)"], label_visibility="collapsed")
    contenido = None
    if opcion.startswith("Subir"):
        archivo = st.file_uploader("CSV con las 13 variables (nombres en inglés o español)", type="csv")
        if archivo is None:
            st.info("Subí un CSV para continuar. Se reconocen las columnas del dataset de Kaggle/UCI.")
            st.stop()
        contenido = archivo.getvalue()
    fuente = {"CSV": "catedra", "Sub": "propio", "Dat": "uci"}[opcion[:3]]
    try:
        df_original, variedad = cargar(fuente, contenido)
    except ValueError as error:
        st.error(f"No se pudo leer el CSV: {error}")
        st.stop()

    variables = st.multiselect("Variables usadas para clusterizar", COLUMNAS, default=COLUMNAS,
                               help="Probá, por ejemplo, quitar Prolina o el bloque fenólico.")
    if len(variables) < 2:
        st.warning("Elegí al menos 2 variables.")
        st.stop()

    st.markdown("### 2 · Preparación")
    escalador_nombre = st.selectbox("Escalado", ESCALADORES)
    winsorizar = st.toggle("Winsorizar outliers (1,5·IQR)", value=False)

    st.markdown("### 3 · Modelo")
    algoritmo = st.selectbox("Algoritmo final", ["K-Means", "Jerárquico aglomerativo"])
    init, n_init, max_iter, linkage, metrica = "k-means++", 10, 300, "ward", "euclidean"
    if algoritmo == "K-Means":
        init = st.selectbox("Inicialización (init)", ["k-means++", "random"])
        n_init = st.slider("n_init (inicializaciones)", 1, 30, 10)
        max_iter = st.slider("max_iter", 10, 500, 300, step=10)
    else:
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        metrica = "euclidean" if linkage == "ward" else st.selectbox("Distancia", ["euclidean", "manhattan",
                                                                                 "chebyshev", "cosine"])
    automatico = st.toggle("Elegir k automáticamente (consenso de criterios)", value=True)
    k_manual = None if automatico else st.slider("k (cantidad de clusters)", 2, 10, 3)
    semilla = int(st.number_input("Semilla (random_state)", 0, 10_000, SEMILLA))

# ---------------------------------------------------------------------------
# Cabecera y solapas
# ---------------------------------------------------------------------------
st.markdown("""<div class="cabecera"><h1>Clustering de vinos tintos</h1>
<p>Aprendizaje no supervisado sobre 13 variables químicas · Aprendizaje Artificial · Maestría en IA (CAECE)</p></div>""",
            unsafe_allow_html=True)
zona_metricas = st.container()
(t_resumen, t_datos, t_prep, t_k, t_modelo, t_alg, t_tipos, t_post) = st.tabs([
    "📋 Resumen", "🔍 Datos y EDA", "⚙️ Preparación", "📈 Elección de k", "🧪 Modelo y estabilidad",
    "🌳 Comparar algoritmos", "🍷 Tipos de vino", "✅ Validación post-hoc"])

# --- Datos (primero, porque la edición alimenta todo lo demás) -------------
with t_datos:
    st.subheader("Datos de trabajo")
    st.caption("La variedad real NO forma parte de estos datos: solo se usa en la solapa de validación post-hoc.")
    with st.expander("✏️ Editar valores (los cambios recalculan todo el análisis)"):
        df = st.data_editor(df_original, num_rows="fixed", width="stretch", height=320, key="editor")
        editado = not df.equals(df_original)
        if editado:
            st.warning("Estás trabajando con datos editados.")
        st.download_button("⬇️ Descargar estos datos (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "vinos_editados.csv", "text/csv")
    df = df.astype(float)

cfg = {"fuente": opcion, "editado": editado, "variables": variables, "escalador": escalador_nombre,
       "winsorizar": winsorizar, "k_manual": k_manual, "semilla": semilla,
       "algoritmo_desc": (f"K-Means (init='{init}', n_init={n_init}, max_iter={max_iter})" if algoritmo == "K-Means"
                          else f"Jerárquico aglomerativo (linkage='{linkage}', distancia {metrica})")}
r = analizar(df, tuple(variables), escalador_nombre, winsorizar, algoritmo, init, n_init, max_iter,
             linkage, metrica, k_manual, semilla)
k, etiquetas, nombres = r["k"], r["etiquetas"], r["nombres"]
FIRMA = str((int(pd.util.hash_pandas_object(df).sum()), tuple(variables), escalador_nombre, winsorizar, algoritmo,
              init, n_init, max_iter, linkage, metrica, k_manual, semilla, variedad is not None))

post = None
if variedad is not None:
    contingencia, metricas = it.validacion_posthoc(etiquetas, variedad)
    post = {"contingencia": contingencia, **metricas}

with zona_metricas:
    c = st.columns(5)
    c[0].metric("Vinos", df.shape[0])
    c[1].metric("Variables usadas", f"{len(variables)} / 13")
    c[2].metric("Clusters (k)", k, help=f"Consenso de criterios: k = {r['k_consenso']}")
    c[3].metric("Silueta", fmt(r["silueta"]))
    c[4].metric("ARI post-hoc", fmt(post["ARI"]) if post else "—",
                help="Acuerdo con la variedad real (solo evaluación, no se usa para aprender).")

with t_datos:
    st.dataframe(eda.resumen_basico(df).style.format(precision=2), width="stretch")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("**Outliers por variable**")
        st.dataframe(eda.contar_outliers(df), width="stretch")
    with col2:
        st.markdown("**Correlaciones fuertes (|r| ≥ 0,5)**")
        st.dataframe(eda.correlaciones_fuertes(eda.matriz_correlacion(df)).style.format({"r": "{:.3f}"}),
                     width="stretch", hide_index=True)
    grafico = st.radio("Gráfico", ["Histogramas", "Boxplots", "Escalas en un mismo eje", "Matriz de correlación",
                                   "Pairplot"], horizontal=True)
    if grafico == "Histogramas":
        mostrar("fig01", lambda: eda.graficar_histogramas(df))
    elif grafico == "Boxplots":
        mostrar("fig02", lambda: eda.graficar_boxplots(df))
    elif grafico == "Escalas en un mismo eje":
        mostrar("fig03", lambda: eda.graficar_boxplots_escala_comun(df))
    elif grafico == "Matriz de correlación":
        mostrar("fig04", lambda: eda.graficar_correlacion(eda.matriz_correlacion(df)))
    else:
        mostrar("fig05", lambda: eda.graficar_pairplot(df, eda.seleccionar_variables_pairplot(df)).figure)

# --- Resumen -----------------------------------------------------------------
with t_resumen:
    izq, der = st.columns([1.35, 1])
    with izq:
        st.subheader("Objetivo")
        st.markdown(
            "Descubrir cuántos **tipos de vino** hay en los datos a partir de su perfil químico, **sin usar la "
            "variedad**, y nombrarlos. Es aprendizaje **no supervisado**: no hay variable objetivo ni train/test; "
            "se buscan grupos con mínima distancia intra-cluster y máxima inter-cluster.")
        st.subheader("Configuración actual")
        st.table(pd.DataFrame({"Elección": [
            f"{opcion}{' (editado)' if editado else ''}", f"{len(variables)} de 13", escalador_nombre,
            "Winsorizados" if winsorizar else "Conservados",
            f"{k} ({'automático' if k_manual is None else 'manual'})", cfg["algoritmo_desc"], str(semilla)]},
            index=["Datos", "Variables", "Escalado", "Outliers", "k", "Algoritmo", "Semilla"]))
    with der:
        st.subheader("Tipos encontrados")
        for cl_id, n in r["tamanos"]["vinos"].items():
            st.markdown(f"<div class='ficha' style='--c:{it.color_cluster(cl_id)}'><b>C{cl_id} · {nombres[cl_id]}</b>"
                        f"<br>{n} vinos ({fmt(r['tamanos'].loc[cl_id, '%'], 1)} %)</div>", unsafe_allow_html=True)
    st.subheader("Lectura automática de los resultados")
    textos = [f"Con esta configuración, los criterios de elección de k sugieren **k = {r['k_consenso']}** "
              f"({', '.join(f'{c}: {v}' for c, v in r['resumen_k']['k sugerido'].items())})."]
    if k_manual is not None and k_manual != r["k_consenso"]:
        textos.append(f"Se forzó manualmente **k = {k}**, distinto del consenso: revisá la silueta por cluster.")
    textos.append(f"El modelo final obtiene una silueta de **{fmt(r['silueta'])}** y tamaños entre "
                  f"{r['tamanos']['vinos'].min()} y {r['tamanos']['vinos'].max()} vinos.")
    if algoritmo != "K-Means":
        textos.append(f"Su acuerdo con K-Means es ARI = {fmt(r['ari_km'])}.")
    if escalador_nombre == "Sin escalar":
        top = r["dominancia"].index[0]
        textos.append(f"⚠️ Sin escalar, **{top}** explica el {fmt(r['dominancia'].iloc[0], 2)} % de la separación "
                      "entre clusters: el agrupamiento queda dominado por una sola variable.")
    if post:
        textos.append(f"Validación post-hoc: ARI = {fmt(post['ARI'])}, pureza = {fmt(100 * post['Pureza'], 1)} %.")
    st.markdown("\n".join(f"- {t}" for t in textos))
    st.download_button("📄 Descargar informe recreado (Markdown)", generar_informe(cfg, r, post, df).encode("utf-8"),
                       "informe_clustering_vinos.md", "text/markdown", type="primary")
    datos_con_cluster = df.assign(cluster=etiquetas, tipo=[nombres[e] for e in etiquetas])
    st.download_button("⬇️ Descargar vinos con su cluster (CSV)", datos_con_cluster.to_csv(index=False).encode("utf-8"),
                       "vinos_con_cluster.csv", "text/csv")

# --- Preparación ---------------------------------------------------------------
with t_prep:
    st.subheader("¿Por qué escalar?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Aporte de cada variable a la distancia² entre los vinos 0 y 1**")
        st.dataframe(eda.ejemplo_dominancia_escala(df, 0, 1).style.format(precision=2), width="stretch")
    with col2:
        part = eda.participacion_varianza(df)
        st.markdown("**% de la varianza total sin escalar**")
        st.bar_chart(part, horizontal=True, color="#7b2d3b")
        st.caption(f"{part.index[0]} concentra el {fmt(part.iloc[0], 1)} % de la varianza: sin escalar, "
                   "K-Means (que minimiza varianzas) agruparía casi solo por esa variable.")
    st.subheader("Comparación de escaladores")
    st.dataframe(pp.comparar_escaladores(df[variables], k=k, semilla=semilla), width="stretch")
    st.caption("Las siluetas se miden en el espacio de cada escalador: son orientativas, no estrictamente comparables.")

# --- Elección de k -------------------------------------------------------------
with t_k:
    st.subheader("Criterios para elegir k")
    st.dataframe(r["resumen_k"], width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        mostrar("fig06", lambda: cl.graficar_codo(r["tabla_k"], r["k_codo"]))
    with col2:
        mostrar("fig07", lambda: cl.graficar_silueta_promedio(r["tabla_k"], int(r["tabla_k"]["silueta"].idxmax())))
    st.dataframe(r["tabla_k"].style.format(precision=3), width="stretch")
    ks = [x for x in (k - 1, k, k + 1) if 2 <= x <= 10]
    mostrar("fig08", lambda: cl.graficar_siluetas_por_cluster(r["X"], ks, semilla))
    k_d = int(r["saltos"]["salto"].idxmax())
    mostrar("fig09", lambda: cl.graficar_dendrograma(r["Z"], r["saltos"].loc[k_d, "umbral de corte"], k_d,
                                    "Dendrograma jerárquico (Ward) sobre los datos preparados"))

# --- Modelo y estabilidad --------------------------------------------------------
with t_modelo:
    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.subheader("Tamaños de los clusters")
        st.dataframe(r["tamanos"].style.format({"%": "{:.1f}"}), width="stretch")
    with col2:
        st.subheader("Estabilidad entre 10 semillas")
        if r["estabilidad"] is None:
            st.info("El clustering jerárquico es determinístico: no depende de la semilla.")
        else:
            st.dataframe(r["estabilidad"], width="stretch")
            ari_min = r["estabilidad"].iloc[0]["ARI mínimo"]
            (st.success if ari_min >= 0.9 else st.warning)(
                f"ARI mínimo entre corridas = {fmt(ari_min)} "
                f"({'estable' if ari_min >= 0.9 else 'inestable: < 0,9, considerar más inicializaciones'}).")
    st.subheader("K-Means sin escalar vs. escalado")
    st.dataframe(r["sin_esc"], width="stretch")
    st.bar_chart(r["dominancia"], horizontal=True, color="#d98c2b")
    st.subheader("Impacto de los outliers (winsorización)")
    st.json(r["outliers"])

# --- Comparar algoritmos ----------------------------------------------------------
with t_alg:
    st.subheader(f"Jerárquico aglomerativo vs. K-Means (k = {k})")
    st.dataframe(r["jerarquicos"].style.format(precision=3), width="stretch")
    st.markdown("**Tabla de contingencia K-Means vs. Ward**")
    st.dataframe(tabla_str(r["contingencia_ward"]), width="stretch")
    st.caption("Linkage simple y medio suelen aislar outliers (tamaños muy desparejos) por el efecto cadena; "
               "Ward es el análogo jerárquico de K-Means.")

# --- Tipos de vino -----------------------------------------------------------------
with t_tipos:
    mostrar("fig10", lambda: it.graficar_heatmap_perfiles(r["perfiles"], nombres))
    for cl_id, f in r["fichas"].items():
        with st.container():
            st.markdown(f"<div class='ficha' style='--c:{it.color_cluster(cl_id)}'><h4>C{cl_id} · {f['nombre']}</h4>"
                        f"{f['descripción']}<br><br><b>Perfil químico:</b><ul>"
                        + "".join(f"<li>{q}</li>" for q in f["perfil químico"])
                        + f"</ul><b>Perfil sensorial (inferencia):</b> {f['perfil sensorial (inferido)']}<br>"
                        f"<b>Uso comercial (inferencia):</b> {f['uso comercial (inferido)']}</div>",
                        unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Centroides (unidades originales)**")
        st.dataframe(r["centroides"].T.style.format(precision=2), width="stretch")
    with col2:
        st.markdown("**Variables más discriminantes (F de ANOVA)**")
        st.dataframe(r["ranking"][["ranking", "F de ANOVA"]].style.format({"F de ANOVA": "{:.1f}"}),
                     width="stretch")
    mostrar("fig11", lambda: it.graficar_boxplots_por_cluster(df, etiquetas, r["ranking"].index[:6].tolist(), nombres))
    mostrar("fig12", lambda: it.graficar_pca(r["pca"], r["proy"], r["loadings"], etiquetas, r["centros_X"], nombres,
                            "Vinos en el plano PCA (solo visualización), coloreados por cluster"))

# --- Validación post-hoc ---------------------------------------------------------------
with t_post:
    st.warning("Esta sección **no forma parte del aprendizaje**: la variedad real no interviene en ninguna decisión "
               "y en un caso real no estaría disponible.")
    if post is None:
        st.info("Los datos actuales no tienen variedad real asociada (el CSV no trae etiqueta y no coincide con UCI).")
    else:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.dataframe(tabla_str(post["contingencia"]), width="stretch")
        with col2:
            m = st.columns(3)
            m[0].metric("ARI", fmt(post["ARI"]))
            m[1].metric("NMI", fmt(post["NMI"]))
            m[2].metric("Pureza", f"{fmt(100 * post['Pureza'], 1)} %")
            st.caption(f"{post['Vinos fuera de la variedad mayoritaria']} vinos quedan en un cluster cuya variedad "
                       "mayoritaria no es la suya.")
        mostrar("fig13", lambda: it.graficar_pca_clusters_vs_real(r["proy"], etiquetas, variedad, r["pca"]))
