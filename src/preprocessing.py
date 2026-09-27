"""Preprocesamiento: comparación de escaladores, escalado final y alternativa de tratamiento de outliers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .data import DIR_PROCESSED, SEMILLA


def construir_escaladores() -> dict[str, object]:
    """Escaladores candidatos a comparar."""
    return {
        "StandardScaler (z-score)": StandardScaler(),
        "MinMaxScaler [0, 1]": MinMaxScaler(),
        "RobustScaler (mediana/IQR)": RobustScaler(),
    }


def escalar(df: pd.DataFrame, escalador=None) -> tuple[pd.DataFrame, object]:
    """Ajusta el escalador (StandardScaler por defecto) y devuelve ``(df_escalado, escalador_ajustado)``."""
    escalador = escalador if escalador is not None else StandardScaler()
    valores = escalador.fit_transform(df)
    return pd.DataFrame(valores, columns=df.columns, index=df.index), escalador


def comparar_escaladores(df: pd.DataFrame, k: int = 3, semilla: int = SEMILLA) -> pd.DataFrame:
    """Compara escaladores: homogeneidad de escalas resultante y calidad de un K-Means exploratorio.

    Para cada opción (incluida "sin escalar") informa el rango de desvíos estándar entre
    variables (cuanto más cercano a 1 el cociente máx/mín, más comparables son), el
    máximo valor absoluto (sensibilidad a outliers), la silueta de un K-Means con ``k``
    clusters y los tamaños obtenidos. La silueta se calcula en el espacio de cada
    escalador, por lo que es orientativa y no una comparación estricta.
    """
    opciones = {"Sin escalar": None, **construir_escaladores()}
    filas = []
    for nombre, escalador in opciones.items():
        X = df.to_numpy() if escalador is None else escalador.fit_transform(df)
        desvios = X.std(axis=0)
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit_predict(X)
        filas.append({
            "escalado": nombre,
            "desvío mín.": desvios.min(),
            "desvío máx.": desvios.max(),
            "cociente máx/mín de desvíos": desvios.max() / desvios.min(),
            "|valor| máximo": np.abs(X).max(),
            f"silueta K-Means k={k}": silhouette_score(X, etiquetas),
            "tamaños": sorted(np.bincount(etiquetas).tolist(), reverse=True),
        })
    return pd.DataFrame(filas).set_index("escalado")


def recortar_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Alternativa documentada (no adoptada): winsorización de cada variable a [Q1 − f·IQR, Q3 + f·IQR]."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    return df.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr, axis=1)


def guardar_procesado(df_escalado: pd.DataFrame, directorio: Path = DIR_PROCESSED,
                      nombre: str = "vinos_escalado.csv") -> Path:
    """Guarda el dataset escalado en ``data/processed``."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre
    df_escalado.to_csv(ruta, index=False, encoding="utf-8")
    return ruta
