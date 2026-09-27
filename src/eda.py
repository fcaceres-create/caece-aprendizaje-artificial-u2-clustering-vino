"""Análisis exploratorio de datos (EDA): resúmenes, outliers, correlaciones y gráficos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Primitivas gráficas (matplotlib puro, compatibles con matplotlib >= 3.10)
# ---------------------------------------------------------------------------
def dibujar_boxplot(eje: plt.Axes, grupos: list, etiquetas: list[str], colores) -> None:
    """Boxplot vertical de varios grupos sobre un eje, con un color por caja."""
    colores = colores if isinstance(colores, list) else [colores] * len(grupos)
    cajas = eje.boxplot(grupos, orientation="vertical", patch_artist=True, widths=0.6,
                        medianprops={"color": "black"},
                        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "white"})
    for caja, color in zip(cajas["boxes"], colores):
        caja.set_facecolor(color)
    eje.set_xticks(range(1, len(etiquetas) + 1), etiquetas)


def dibujar_heatmap(eje: plt.Axes, datos: pd.DataFrame, vmin: float, vmax: float, etiqueta_barra: str,
                    mascara: np.ndarray | None = None) -> None:
    """Heatmap anotado con escala divergente centrada en 0 (celdas enmascaradas en blanco)."""
    valores = datos.to_numpy(dtype=float)
    visibles = np.ma.masked_array(valores, mask=mascara if mascara is not None else np.zeros_like(valores, bool))
    imagen = eje.imshow(visibles, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(valores):
        if mascara is None or not mascara[i, j]:
            eje.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(v) > 0.6 * max(abs(vmin), vmax) else "black")
    eje.set_xticks(range(datos.shape[1]), datos.columns, rotation=40, ha="right")
    eje.set_yticks(range(datos.shape[0]), datos.index)
    eje.figure.colorbar(imagen, ax=eje, label=etiqueta_barra)


def resumen_basico(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla por variable con tipo de dato, nulos, mínimo, máximo, media y desvío."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "mínimo": df.min(),
        "máximo": df.max(),
        "media": df.mean(),
        "desvío": df.std(),
        "coef. variación": df.std() / df.mean(),
    })


def contar_outliers(df: pd.DataFrame, umbral_z: float = 3.0) -> pd.DataFrame:
    """Cuenta outliers por variable según el criterio IQR (1,5·IQR) y según |z| > ``umbral_z``."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    fuera_iqr = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
    z = (df - df.mean()) / df.std()
    tabla = pd.DataFrame({
        "outliers IQR": fuera_iqr.sum(),
        f"outliers |z|>{umbral_z:g}": (z.abs() > umbral_z).sum(),
    })
    tabla.loc["Total (filas distintas)"] = [fuera_iqr.any(axis=1).sum(), (z.abs() > umbral_z).any(axis=1).sum()]
    return tabla


def graficar_histogramas(df: pd.DataFrame) -> plt.Figure:
    """Histogramas con curva de densidad de las 13 variables (una por panel)."""
    fig, ejes = plt.subplots(4, 4, figsize=(16, 13))
    for eje, col in zip(ejes.flat, df.columns):
        sns.histplot(df[col], kde=True, ax=eje, color="#7b2d3b")
        eje.set_title(col)
        eje.set_xlabel("Valor")
        eje.set_ylabel("Frecuencia")
    for eje in ejes.flat[len(df.columns):]:
        eje.axis("off")
    fig.suptitle("Distribución de las 13 variables químicas (178 vinos)", fontsize=15)
    fig.tight_layout()
    return fig


def graficar_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Boxplots individuales (cada variable con su propia escala) para detectar outliers."""
    fig, ejes = plt.subplots(2, 7, figsize=(20, 8))
    for eje, col in zip(ejes.flat, df.columns):
        dibujar_boxplot(eje, [df[col]], [""], "#c98b96")
        eje.set_title(col, fontsize=10)
        eje.set_ylabel("Valor (unidades originales)")
    ejes.flat[-1].axis("off")
    fig.suptitle("Boxplots por variable: escalas muy distintas y presencia de outliers", fontsize=15)
    fig.tight_layout()
    return fig


def graficar_boxplots_escala_comun(df: pd.DataFrame) -> plt.Figure:
    """Boxplots de todas las variables en un mismo eje: evidencia visual de las diferencias de escala."""
    fig, eje = plt.subplots(figsize=(14, 6))
    dibujar_boxplot(eje, [df[c] for c in df.columns], list(df.columns), "#c98b96")
    eje.set_yscale("log")
    eje.set_title("Todas las variables en un mismo eje (escala logarítmica): Prolina y Magnesio dominan")
    eje.set_xlabel("Variable")
    eje.set_ylabel("Valor original (escala log)")
    eje.tick_params(axis="x", rotation=45)
    for etiqueta in eje.get_xticklabels():
        etiqueta.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación de Pearson."""
    return df.corr(method="pearson")


def correlaciones_fuertes(corr: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    """Pares de variables con |r| >= ``umbral``, ordenados por magnitud."""
    pares = []
    columnas = corr.columns
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            r = corr.iloc[i, j]
            if abs(r) >= umbral:
                pares.append({"variable 1": columnas[i], "variable 2": columnas[j], "r": r})
    tabla = pd.DataFrame(pares)
    return tabla.reindex(tabla["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def graficar_correlacion(corr: pd.DataFrame) -> plt.Figure:
    """Heatmap de la matriz de correlación con escala divergente centrada en 0."""
    fig, eje = plt.subplots(figsize=(12, 10))
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    dibujar_heatmap(eje, corr, -1, 1, "Correlación de Pearson (r)", mascara)
    eje.set_title("Matriz de correlación entre variables químicas")
    eje.set_xlabel("Variable")
    eje.set_ylabel("Variable")
    fig.tight_layout()
    return fig


def seleccionar_variables_pairplot(df: pd.DataFrame, n: int = 6) -> list[str]:
    """Elige las ``n`` variables con mayor coeficiente de variación (dispersión relativa, sin usar etiquetas)."""
    cv = df.std() / df.mean()
    return cv.sort_values(ascending=False).head(n).index.tolist()


def graficar_pairplot(df: pd.DataFrame, variables: list[str]) -> sns.PairGrid:
    """Pairplot (dispersión por pares + densidades en la diagonal) de un subconjunto de variables."""
    grilla = sns.pairplot(df[variables], diag_kind="kde", corner=True,
                          plot_kws={"s": 18, "alpha": 0.7, "color": "#7b2d3b"},
                          diag_kws={"color": "#7b2d3b"})
    grilla.figure.suptitle("Pairplot de las variables con mayor dispersión relativa", y=1.01, fontsize=14)
    return grilla


def ejemplo_dominancia_escala(df: pd.DataFrame, i: int = 0, j: int = 1) -> pd.DataFrame:
    """Aporte de cada variable a la distancia euclídea² entre los vinos ``i`` y ``j``, sin y con escalado.

    Demuestra numéricamente que, sin escalar, Prolina (y en menor medida Magnesio)
    determinan casi por completo la distancia entre dos vinos.
    """
    escalado = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    dif_orig = (df.iloc[i] - df.iloc[j]) ** 2
    dif_esc = (escalado.iloc[i] - escalado.iloc[j]) ** 2
    tabla = pd.DataFrame({
        f"vino {i}": df.iloc[i],
        f"vino {j}": df.iloc[j],
        "dif² sin escalar": dif_orig,
        "% de la distancia² sin escalar": 100 * dif_orig / dif_orig.sum(),
        "dif² escalado (z)": dif_esc,
        "% de la distancia² escalado": 100 * dif_esc / dif_esc.sum(),
    })
    tabla.loc["TOTAL"] = [np.nan, np.nan, dif_orig.sum(), 100.0, dif_esc.sum(), 100.0]
    return tabla


def participacion_varianza(df: pd.DataFrame) -> pd.Series:
    """Porcentaje de la varianza total (suma de varianzas) que aporta cada variable sin escalar.

    Como la inercia de K-Means es una suma de varianzas, esta proporción indica qué
    variables dominarían el agrupamiento sobre datos crudos.
    """
    var = df.var()
    return (100 * var / var.sum()).sort_values(ascending=False).rename("% varianza total")
