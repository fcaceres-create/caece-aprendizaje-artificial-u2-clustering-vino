"""Elección de k, K-Means, estabilidad entre corridas y clustering jerárquico aglomerativo."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)

from .data import SEMILLA


# ---------------------------------------------------------------------------
# K-Means: construcción del modelo
# ---------------------------------------------------------------------------
def crear_kmeans(k: int, semilla: int = SEMILLA, n_init: int = 10, init: str = "k-means++",
                 max_iter: int = 300) -> KMeans:
    """K-Means con los hiperparámetros del trabajo (k-means++, 10 inicializaciones, 300 iteraciones)."""
    return KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=semilla)


# ---------------------------------------------------------------------------
# Elección de k
# ---------------------------------------------------------------------------
def evaluar_rango_k(X: np.ndarray, ks=range(1, 11), semilla: int = SEMILLA) -> pd.DataFrame:
    """Inercia (WCSS), reducción porcentual, silueta, Calinski-Harabasz, Davies-Bouldin y tamaños para cada k."""
    filas = []
    for k in ks:
        modelo = crear_kmeans(k, semilla).fit(X)
        etiquetas = modelo.labels_
        conteos = np.bincount(etiquetas)
        fila = {"k": k, "inercia (WCSS)": modelo.inertia_,
                "tamaño mín.": conteos.min(), "tamaño máx.": conteos.max()}
        if k > 1:
            fila["silueta"] = silhouette_score(X, etiquetas)
            fila["Calinski-Harabasz"] = calinski_harabasz_score(X, etiquetas)
            fila["Davies-Bouldin"] = davies_bouldin_score(X, etiquetas)
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("k")
    tabla.insert(1, "reducción % vs k-1", -100 * tabla["inercia (WCSS)"].pct_change())
    return tabla


def detectar_codo(inercias: pd.Series) -> int:
    """Detecta el codo como el punto de la curva más alejado de la recta que une sus extremos.

    Se normalizan ambos ejes a [0, 1] para que la distancia no dependa de las unidades.
    """
    k = np.asarray(inercias.index, dtype=float)
    y = inercias.to_numpy(dtype=float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # Distancia de cada punto a la recta que une (0, 1) con (1, 0): |x + y − 1| / √2.
    distancias = np.abs(kn + yn - 1) / np.sqrt(2)
    return int(k[np.argmax(distancias)])


def graficar_codo(tabla: pd.DataFrame, k_codo: int) -> plt.Figure:
    """Curva del método del codo con el codo marcado y la reducción porcentual anotada."""
    fig, eje = plt.subplots(figsize=(9, 5.5))
    inercia = tabla["inercia (WCSS)"]
    eje.plot(inercia.index, inercia.values, marker="o", color="#7b2d3b", label="Inercia (WCSS)")
    eje.scatter([k_codo], [inercia[k_codo]], s=250, facecolors="none", edgecolors="black",
                linewidths=2, zorder=3, label=f"Codo detectado (k = {k_codo})")
    for k, red in tabla["reducción % vs k-1"].dropna().items():
        eje.annotate(f"−{red:.1f}%", (k, inercia[k]), textcoords="offset points", xytext=(6, 8), fontsize=8)
    eje.set_title("Método del codo: inercia (WCSS) de K-Means según k")
    eje.set_xlabel("Cantidad de clusters (k)")
    eje.set_ylabel("Inercia / WCSS (suma de distancias² al centroide)")
    eje.set_xticks(list(inercia.index))
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    return fig


def graficar_silueta_promedio(tabla: pd.DataFrame, k_elegido: int) -> plt.Figure:
    """Coeficiente de silueta promedio para k = 2..10."""
    fig, eje = plt.subplots(figsize=(9, 5.5))
    silueta = tabla["silueta"].dropna()
    eje.plot(silueta.index, silueta.values, marker="o", color="#2d5f7b", label="Silueta promedio")
    eje.scatter([k_elegido], [silueta[k_elegido]], s=250, facecolors="none", edgecolors="black",
                linewidths=2, zorder=3, label=f"Máximo (k = {k_elegido}; s = {silueta[k_elegido]:.3f})")
    eje.set_title("Coeficiente de silueta (silhouette) promedio según k")
    eje.set_xlabel("Cantidad de clusters (k)")
    eje.set_ylabel("Silueta promedio")
    eje.set_xticks(list(silueta.index))
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    return fig


def graficar_siluetas_por_cluster(X: np.ndarray, ks: list[int], semilla: int = SEMILLA) -> plt.Figure:
    """Gráfico de silueta por muestra, agrupado por cluster, para varios k (uno por panel)."""
    fig, ejes = plt.subplots(1, len(ks), figsize=(6 * len(ks), 6), sharex=True)
    for eje, k in zip(np.atleast_1d(ejes), ks):
        etiquetas = crear_kmeans(k, semilla).fit_predict(X)
        valores = silhouette_samples(X, etiquetas)
        promedio = valores.mean()
        y_inf = 10
        for c in range(k):
            vc = np.sort(valores[etiquetas == c])
            y_sup = y_inf + len(vc)
            color = cm.tab10(c)
            eje.fill_betweenx(np.arange(y_inf, y_sup), 0, vc, facecolor=color, alpha=0.8,
                              label=f"Cluster {c} (n={len(vc)})")
            y_inf = y_sup + 10
        eje.axvline(promedio, color="red", linestyle="--", label=f"Promedio = {promedio:.3f}")
        eje.set_title(f"k = {k}")
        eje.set_xlabel("Coeficiente de silueta")
        eje.set_ylabel("Muestras (ordenadas por cluster)")
        eje.set_yticks([])
        eje.legend(fontsize=8, loc="lower right")
    fig.suptitle("Silueta por muestra y por cluster (K-Means)", fontsize=14)
    fig.tight_layout()
    return fig


def calcular_linkage(X: np.ndarray, metodo: str = "ward", metrica: str = "euclidean") -> np.ndarray:
    """Matriz de enlace (linkage) de SciPy para el dendrograma."""
    return linkage(X, method=metodo, metric=metrica)


def saltos_dendrograma(Z: np.ndarray, k_max: int = 10) -> pd.DataFrame:
    """Salto de distancia de fusión asociado a cortar el dendrograma en k clusters (k = 2..k_max).

    Con alturas de fusión ordenadas d₁ ≤ … ≤ d_{n−1}, cortar entre d_{n−k} y d_{n−k+1}
    deja k clusters; el salto es la diferencia entre ambas alturas. Cuanto mayor el salto,
    más "natural" es ese corte.
    """
    alturas = np.sort(Z[:, 2])
    n = len(alturas) + 1
    filas = []
    for k in range(2, k_max + 1):
        inferior, superior = alturas[n - k - 1], alturas[n - k]
        filas.append({"k": k, "altura inferior": inferior, "altura superior": superior,
                      "salto": superior - inferior, "umbral de corte": (inferior + superior) / 2})
    return pd.DataFrame(filas).set_index("k")


def graficar_dendrograma(Z: np.ndarray, umbral: float, k: int, titulo: str) -> plt.Figure:
    """Dendrograma truncado (últimas 40 fusiones) con la línea de corte sugerida."""
    fig, eje = plt.subplots(figsize=(13, 6))
    dendrogram(Z, truncate_mode="lastp", p=40, color_threshold=umbral, ax=eje,
               leaf_rotation=90, leaf_font_size=8, show_contracted=True)
    eje.axhline(umbral, color="black", linestyle="--", label=f"Corte sugerido → {k} clusters")
    eje.set_title(titulo)
    eje.set_xlabel("Vinos o grupos de vinos (entre paréntesis, cantidad de vinos)")
    eje.set_ylabel("Distancia de fusión")
    eje.legend()
    fig.tight_layout()
    return fig


def resumen_eleccion_k(tabla: pd.DataFrame, k_codo: int, saltos: pd.DataFrame) -> pd.DataFrame:
    """Tabla 'criterio vs. k sugerido' para fundamentar la decisión final."""
    candidatos = tabla.dropna(subset=["silueta"])
    filas = [
        ("Método del codo (WCSS)", k_codo, f"reducción de {tabla.loc[k_codo, 'reducción % vs k-1']:.1f}% "
         f"al pasar a k={k_codo} y de {tabla.loc[k_codo + 1, 'reducción % vs k-1']:.1f}% al siguiente"),
        ("Silueta promedio (máx.)", int(candidatos["silueta"].idxmax()),
         f"s = {candidatos['silueta'].max():.3f}"),
        ("Calinski-Harabasz (máx.)", int(candidatos["Calinski-Harabasz"].idxmax()),
         f"CH = {candidatos['Calinski-Harabasz'].max():.1f}"),
        ("Davies-Bouldin (mín.)", int(candidatos["Davies-Bouldin"].idxmin()),
         f"DB = {candidatos['Davies-Bouldin'].min():.3f}"),
        ("Mayor salto del dendrograma (Ward)", int(saltos["salto"].idxmax()),
         f"salto = {saltos['salto'].max():.2f}"),
    ]
    return pd.DataFrame(filas, columns=["criterio", "k sugerido", "evidencia"]).set_index("criterio")


# ---------------------------------------------------------------------------
# Estabilidad y sanidad
# ---------------------------------------------------------------------------
def estabilidad_semillas(X: np.ndarray, k: int, semillas=range(10), n_init: int = 10,
                         init: str = "k-means++") -> dict:
    """Corre K-Means con varias semillas y mide el acuerdo entre corridas con el ARI por pares."""
    corridas = [crear_kmeans(k, s, n_init=n_init, init=init).fit(X) for s in semillas]
    aris = [adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(corridas, 2)]
    inercias = [m.inertia_ for m in corridas]
    return {
        "configuración": f"init='{init}', n_init={n_init}",
        "corridas": len(corridas),
        "ARI medio": float(np.mean(aris)),
        "ARI mínimo": float(np.min(aris)),
        "inercia mín.": float(np.min(inercias)),
        "inercia máx.": float(np.max(inercias)),
        "soluciones distintas": len({round(i, 6) for i in inercias}),
    }


def tamanos_clusters(etiquetas: np.ndarray, nombres=None) -> pd.DataFrame:
    """Conteo y porcentaje de vinos por cluster."""
    conteo = pd.Series(etiquetas).value_counts().sort_index()
    if nombres is not None:
        conteo.index = [nombres.get(c, c) for c in conteo.index]
    return pd.DataFrame({"vinos": conteo, "%": 100 * conteo / conteo.sum()}).rename_axis("cluster")


def comparar_sin_escalar(df: pd.DataFrame, X_escalado: np.ndarray, etiquetas_ref: np.ndarray,
                         k: int, semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.Series]:
    """K-Means sobre datos crudos vs. escalados: tamaños, siluetas, ARI y variables que dominan.

    Devuelve ``(tabla_comparativa, dominancia)`` donde ``dominancia`` es el porcentaje de la
    suma de cuadrados entre clusters (separación) que explica cada variable en la solución
    sin escalar.
    """
    crudo = df.to_numpy()
    et_crudo = crear_kmeans(k, semilla).fit_predict(crudo)
    tabla = pd.DataFrame({
        "tamaños": [sorted(np.bincount(et_crudo).tolist(), reverse=True),
                    sorted(np.bincount(etiquetas_ref).tolist(), reverse=True)],
        "silueta (espacio original)": [silhouette_score(crudo, et_crudo), silhouette_score(crudo, etiquetas_ref)],
        "silueta (espacio escalado)": [silhouette_score(X_escalado, et_crudo),
                                       silhouette_score(X_escalado, etiquetas_ref)],
        "ARI vs. K-Means escalado": [adjusted_rand_score(etiquetas_ref, et_crudo), 1.0],
    }, index=["K-Means SIN escalar", "K-Means CON StandardScaler"])
    medias = df.groupby(et_crudo).mean()
    tamanos = np.bincount(et_crudo)
    ss_entre = ((medias - df.mean()) ** 2).mul(tamanos, axis=0).sum()
    dominancia = (100 * ss_entre / ss_entre.sum()).sort_values(ascending=False)
    return tabla, dominancia.rename("% de la separación entre clusters (sin escalar)")


def impacto_outliers(df_winsorizado_escalado: np.ndarray, etiquetas_ref: np.ndarray, k: int,
                     semilla: int = SEMILLA) -> dict:
    """Compara K-Means con outliers conservados vs. winsorizados (ARI y vinos que cambian de cluster)."""
    et = crear_kmeans(k, semilla).fit_predict(df_winsorizado_escalado)
    contingencia = pd.crosstab(etiquetas_ref, et).to_numpy()
    # Emparejamiento óptimo de clusters (algoritmo húngaro): los IDs son arbitrarios.
    filas, columnas = linear_sum_assignment(-contingencia)
    coincidencias = contingencia[filas, columnas].sum()
    return {"ARI vs. solución con outliers": adjusted_rand_score(etiquetas_ref, et),
            "vinos que cambian de cluster": int(len(et) - coincidencias),
            "tamaños con winsorización": sorted(np.bincount(et).tolist(), reverse=True)}


# ---------------------------------------------------------------------------
# Clustering jerárquico aglomerativo
# ---------------------------------------------------------------------------
CONFIGURACIONES_JERARQUICAS = [
    ("ward", "euclidean"),
    ("complete", "euclidean"),
    ("average", "euclidean"),
    ("single", "euclidean"),
    ("average", "manhattan"),
    ("complete", "manhattan"),
]


def comparar_jerarquicos(X: np.ndarray, k: int, etiquetas_kmeans: np.ndarray,
                         configuraciones=CONFIGURACIONES_JERARQUICAS) -> tuple[pd.DataFrame, dict]:
    """Ajusta ``AgglomerativeClustering`` con varios linkages/métricas y los compara con K-Means.

    Devuelve ``(tabla, etiquetas_por_modelo)``. La silueta se informa con distancia euclídea
    (comparable entre modelos) y con la métrica propia de cada configuración.
    """
    filas, etiquetas = [], {}
    filas.append({"modelo": "K-Means (k-means++)", "linkage": "—", "métrica": "euclidean",
                  "silueta (euclídea)": silhouette_score(X, etiquetas_kmeans),
                  "silueta (métrica propia)": silhouette_score(X, etiquetas_kmeans),
                  "tamaños": sorted(np.bincount(etiquetas_kmeans).tolist(), reverse=True),
                  "ARI vs. K-Means": 1.0})
    for metodo, metrica in configuraciones:
        modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo, metric=metrica)
        et = modelo.fit_predict(X)
        nombre = f"Jerárquico {metodo} ({metrica})"
        etiquetas[nombre] = et
        filas.append({"modelo": nombre, "linkage": metodo, "métrica": metrica,
                      "silueta (euclídea)": silhouette_score(X, et),
                      "silueta (métrica propia)": silhouette_score(X, et, metric=metrica),
                      "tamaños": sorted(np.bincount(et).tolist(), reverse=True),
                      "ARI vs. K-Means": adjusted_rand_score(etiquetas_kmeans, et)})
    return pd.DataFrame(filas).set_index("modelo"), etiquetas


def tabla_contingencia(filas: np.ndarray, columnas: np.ndarray, nombre_filas: str,
                       nombre_columnas: str) -> pd.DataFrame:
    """Tabla de contingencia entre dos particiones (con totales)."""
    return pd.crosstab(pd.Series(filas, name=nombre_filas), pd.Series(columnas, name=nombre_columnas),
                       margins=True, margins_name="Total")
