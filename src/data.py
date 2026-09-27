"""Carga del dataset de vinos, renombrado de columnas al español y utilidades de E/S.

Si existe un CSV en ``data/raw/`` se usa ese archivo (tolerando nombres de columna
en inglés o español, con o sin mayúsculas/acentos). Si no, se carga
``sklearn.datasets.load_wine()`` y la variedad real (``target``) se separa y se
guarda en ``data/raw/etiquetas_ocultas.csv``: NO se usa en ningún paso del análisis,
solo en la validación post-hoc final.
"""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
SEMILLA = 42

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "outputs" / "figures"
DIR_TABLAS = RAIZ / "outputs" / "tables"
ARCHIVO_ETIQUETAS = DIR_RAW / "etiquetas_ocultas.csv"

COLUMNAS = [
    "Alcohol",
    "Ácido málico",
    "Ceniza",
    "Alcalinidad de la ceniza",
    "Magnesio",
    "Fenoles totales",
    "Flavonoides",
    "Fenoles no flavonoides",
    "Proantocianinas",
    "Intensidad del color",
    "Tono",
    "OD280/OD315",
    "Prolina",
]

# Mapeo sklearn -> español (nombres oficiales de load_wine()).
MAPEO_SKLEARN = {
    "alcohol": "Alcohol",
    "malic_acid": "Ácido málico",
    "ash": "Ceniza",
    "alcalinity_of_ash": "Alcalinidad de la ceniza",
    "magnesium": "Magnesio",
    "total_phenols": "Fenoles totales",
    "flavanoids": "Flavonoides",
    "nonflavanoid_phenols": "Fenoles no flavonoides",
    "proanthocyanins": "Proantocianinas",
    "color_intensity": "Intensidad del color",
    "hue": "Tono",
    "od280/od315_of_diluted_wines": "OD280/OD315",
    "proline": "Prolina",
}

# Variantes habituales de nombres (UCI, Kaggle "wine-clustering", español).
# Las claves se comparan ya normalizadas (minúsculas, sin acentos ni símbolos).
_ALIAS = {
    "Alcohol": ["alcohol"],
    "Ácido málico": ["malicacid", "malic", "acidomalico"],
    "Ceniza": ["ash", "ceniza", "cenizas"],
    "Alcalinidad de la ceniza": [
        "alcalinityofash", "ashalcanity", "ashalcalinity", "alcalinityash",
        "alcalinidaddelaceniza", "alcalinidadceniza", "alcalinidaddelascenizas",
    ],
    "Magnesio": ["magnesium", "magnesio", "mg"],
    "Fenoles totales": ["totalphenols", "phenols", "fenolestotales"],
    "Flavonoides": ["flavanoids", "flavonoids", "flavonoides", "flavanoides"],
    "Fenoles no flavonoides": [
        "nonflavanoidphenols", "nonflavonoidphenols", "fenolesnoflavonoides",
    ],
    "Proantocianinas": ["proanthocyanins", "proanthocyanidins", "proantocianinas", "proantocianidinas"],
    "Intensidad del color": ["colorintensity", "intensidaddelcolor", "intensidadcolor", "intensidaddecolor"],
    "Tono": ["hue", "tono", "matiz"],
    "OD280/OD315": [
        "od280od315ofdilutedwines", "od280od315", "od280", "od", "od280od315delosvinosdiluidos",
    ],
    "Prolina": ["proline", "prolina"],
}

# Nombres de columna que indican la variedad real (se separan, nunca se usan para aprender).
_ALIAS_ETIQUETA = {"target", "class", "clase", "customersegment", "tipo", "variedad", "cultivar", "wine", "type"}


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fija las semillas de ``random`` y ``numpy`` para garantizar reproducibilidad."""
    random.seed(semilla)
    np.random.seed(semilla)


def normalizar_nombre(nombre: str) -> str:
    """Normaliza un nombre de columna: minúsculas, sin acentos y solo caracteres alfanuméricos."""
    sin_acentos = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    return "".join(c for c in sin_acentos.lower() if c.isalnum())


_INDICE_ALIAS = {alias: destino for destino, lista in _ALIAS.items() for alias in lista}


def mapear_columnas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Renombra las columnas a los 13 nombres en español y separa una posible etiqueta.

    Devuelve ``(atributos, etiqueta)``: ``atributos`` tiene exactamente las 13 columnas
    en el orden de :data:`COLUMNAS`; ``etiqueta`` es la columna de variedad si el CSV la
    traía (o ``None``). Lanza ``ValueError`` si falta algún atributo.
    """
    renombres, etiqueta = {}, None
    for col in df.columns:
        clave = normalizar_nombre(col)
        if clave in _INDICE_ALIAS:
            renombres[col] = _INDICE_ALIAS[clave]
        elif clave in _ALIAS_ETIQUETA:
            etiqueta = df[col].rename("variedad_real")
    faltantes = set(COLUMNAS) - set(renombres.values())
    if faltantes:
        raise ValueError(f"No se pudieron identificar las columnas: {sorted(faltantes)}")
    atributos = df.rename(columns=renombres)[COLUMNAS].astype(float)
    return atributos, etiqueta


def _csv_de_catedra(dir_raw: Path) -> Path | None:
    """Devuelve el primer CSV de ``dir_raw`` que no sea el de etiquetas ocultas (o ``None``)."""
    candidatos = sorted(p for p in dir_raw.glob("*.csv") if p.name != ARCHIVO_ETIQUETAS.name)
    return candidatos[0] if candidatos else None


def cargar_vinos(dir_raw: Path = DIR_RAW) -> pd.DataFrame:
    """Carga el dataset de trabajo (178 × 13) SIN la variedad real.

    Prioriza un CSV provisto por la cátedra en ``data/raw/``; si no hay, usa
    ``load_wine()`` de scikit-learn. En ambos casos, si se dispone de la variedad
    real, se guarda aparte en ``etiquetas_ocultas.csv`` y se descarta del dataframe.
    """
    dir_raw.mkdir(parents=True, exist_ok=True)
    ruta_csv = _csv_de_catedra(dir_raw)
    if ruta_csv is not None:
        atributos, etiqueta = mapear_columnas(pd.read_csv(ruta_csv))
    else:
        wine = load_wine(as_frame=True)
        atributos = wine.data.rename(columns=MAPEO_SKLEARN)[COLUMNAS]
        # La variedad se descarta del dataframe de trabajo (aprendizaje no supervisado).
        etiqueta = (wine.target + 1).rename("variedad_real")  # variedades 1, 2 y 3 como en UCI
    if etiqueta is not None:
        etiqueta.to_frame().to_csv(ARCHIVO_ETIQUETAS, index=False)
    return atributos.reset_index(drop=True)


def cargar_etiquetas_ocultas(ruta: Path = ARCHIVO_ETIQUETAS) -> pd.Series | None:
    """Carga la variedad real guardada aparte. Uso EXCLUSIVO en la validación post-hoc."""
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)["variedad_real"]


# ---------------------------------------------------------------------------
# Utilidades de salida (figuras y tablas)
# ---------------------------------------------------------------------------
def guardar_figura(fig: plt.Figure, nombre: str, directorio: Path = DIR_FIGURAS) -> Path:
    """Guarda una figura en PNG a 150 dpi y devuelve la ruta."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def guardar_tabla(df: pd.DataFrame, nombre: str, directorio: Path = DIR_TABLAS,
                  indice: bool = True, decimales: int = 3) -> Path:
    """Guarda una tabla como CSV y como Markdown (mismo nombre base) y devuelve la ruta del CSV."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta_csv = directorio / f"{nombre}.csv"
    df.to_csv(ruta_csv, index=indice, encoding="utf-8")
    (directorio / f"{nombre}.md").write_text(
        df.round(decimales).to_markdown(index=indice) + "\n", encoding="utf-8"
    )
    return ruta_csv
