"""Interpretación de clusters: orden determinístico, perfiles, variables discriminantes, PCA,
nombres de los tipos de vino y validación post-hoc con las etiquetas ocultas."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .eda import dibujar_boxplot, dibujar_heatmap

PALETA = {1: "#7b2d3b", 2: "#d98c2b", 3: "#4f8fbf"}


# ---------------------------------------------------------------------------
# Orden determinístico de los clusters
# ---------------------------------------------------------------------------
def ordenar_clusters(etiquetas: np.ndarray, df: pd.DataFrame,
                     variable: str = "Prolina") -> tuple[np.ndarray, dict[int, int]]:
    """Renumera los clusters 1..k por media descendente de ``variable`` (por defecto, Prolina).

    Así los IDs no dependen de la inicialización aleatoria de K-Means. Devuelve
    ``(etiquetas_ordenadas, mapeo_original_a_nuevo)``.
    """
    medias = df.groupby(etiquetas)[variable].mean().sort_values(ascending=False)
    mapeo = {int(original): nuevo for nuevo, original in enumerate(medias.index, start=1)}
    return np.vectorize(mapeo.get)(etiquetas), mapeo


def centroides_originales(modelo, escalador, mapeo: dict[int, int], columnas) -> pd.DataFrame:
    """Centroides de K-Means des-escalados a unidades originales (``inverse_transform``), ya renumerados."""
    valores = escalador.inverse_transform(modelo.cluster_centers_)
    tabla = pd.DataFrame(valores, columns=columnas)
    tabla.index = [mapeo[i] for i in range(len(tabla))]
    return tabla.sort_index().rename_axis("cluster")


# ---------------------------------------------------------------------------
# Perfiles y variables discriminantes
# ---------------------------------------------------------------------------
def perfiles_zscore(df_escalado: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Media de cada cluster en unidades z (desvíos respecto de la media global)."""
    return df_escalado.groupby(etiquetas).mean().rename_axis("cluster")


def comparar_con_media_global(df: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Media por cluster en unidades originales junto con la media global (variables en filas)."""
    tabla = df.groupby(etiquetas).mean().T
    tabla.columns = [f"Cluster {c}" for c in tabla.columns]
    tabla["Media global"] = df.mean()
    return tabla


def graficar_heatmap_perfiles(perfiles: pd.DataFrame, nombres: dict[int, str] | None = None) -> plt.Figure:
    """Heatmap clusters × variables en z-score, con escala divergente centrada en 0."""
    datos = perfiles.copy()
    if nombres:
        datos.index = [f"C{c} · {nombres[c]}" for c in datos.index]
    fig, eje = plt.subplots(figsize=(15, 4.5))
    dibujar_heatmap(eje, datos, -1.5, 1.5, "z-score del centroide")
    eje.set_title("Perfil químico de cada cluster (media del cluster en desvíos respecto de la media global)")
    eje.set_xlabel("Variable")
    eje.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def ranking_discriminantes(df_escalado: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Ranking de variables por su poder de separación entre clusters.

    Usa el estadístico F de ANOVA de un factor (varianza entre clusters / varianza dentro)
    y, como segunda medida, la varianza de las medias de cluster en unidades z.
    """
    f, p = f_classif(df_escalado, etiquetas)
    var_medias = df_escalado.groupby(etiquetas).mean().var()
    tabla = pd.DataFrame({"F de ANOVA": f, "p-valor": p, "varianza de medias (z)": var_medias.values},
                         index=df_escalado.columns)
    tabla = tabla.sort_values("F de ANOVA", ascending=False)
    tabla.insert(0, "ranking", range(1, len(tabla) + 1))
    return tabla.rename_axis("variable")


def graficar_boxplots_por_cluster(df: pd.DataFrame, etiquetas: np.ndarray, variables: list[str],
                                  nombres: dict[int, str] | None = None) -> plt.Figure:
    """Boxplots de las variables indicadas segmentados por cluster (unidades originales)."""
    clusters = sorted(np.unique(etiquetas))
    fig, ejes = plt.subplots(2, 3, figsize=(16, 9))
    for eje, var in zip(ejes.flat, variables):
        dibujar_boxplot(eje, [df.loc[etiquetas == c, var] for c in clusters], [f"C{c}" for c in clusters],
                        [PALETA[c] for c in clusters])
        eje.set_title(var)
        eje.set_xlabel("Cluster")
        eje.set_ylabel(f"{var} (unidades originales)")
    if nombres:
        manijas = [plt.Rectangle((0, 0), 1, 1, color=PALETA[c]) for c in sorted(nombres)]
        fig.legend(manijas, [f"C{c}: {nombres[c]}" for c in sorted(nombres)], loc="lower center",
                   ncol=len(nombres), frameon=False, fontsize=11)
    fig.suptitle("Las 6 variables más discriminantes, por cluster", fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# PCA (solo visualización)
# ---------------------------------------------------------------------------
def ajustar_pca(df_escalado: pd.DataFrame, n: int = 2) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """PCA sobre los datos escalados. Devuelve ``(pca, proyecciones, loadings)``."""
    pca = PCA(n_components=n, random_state=0).fit(df_escalado)
    columnas = [f"PC{i + 1}" for i in range(n)]
    proyecciones = pd.DataFrame(pca.transform(df_escalado), columns=columnas, index=df_escalado.index)
    loadings = pd.DataFrame(pca.components_.T, index=df_escalado.columns, columns=columnas).rename_axis("variable")
    return pca, proyecciones, loadings


def graficar_pca(pca: PCA, proyecciones: pd.DataFrame, loadings: pd.DataFrame, etiquetas: np.ndarray,
                 centroides_escalados: np.ndarray, nombres: dict[int, str], titulo: str) -> plt.Figure:
    """Biplot: vinos coloreados por cluster, centroides proyectados y flechas de loadings."""
    var = pca.explained_variance_ratio_ * 100
    fig, eje = plt.subplots(figsize=(12, 9))
    for c in sorted(np.unique(etiquetas)):
        m = etiquetas == c
        eje.scatter(proyecciones.loc[m, "PC1"], proyecciones.loc[m, "PC2"], s=40, alpha=0.75,
                    color=PALETA.get(c), label=f"C{c}: {nombres.get(c, c)} (n={m.sum()})")
    if centroides_escalados is not None:
        cp = pca.transform(pd.DataFrame(centroides_escalados, columns=loadings.index))
        eje.scatter(cp[:, 0], cp[:, 1], s=350, marker="X", color="black", edgecolor="white",
                    linewidths=1.5, label="Centroides", zorder=4)
    escala = 0.9 * np.abs(proyecciones.to_numpy()).max() / np.abs(loadings.to_numpy()).max()
    for var_nombre, fila in loadings.iterrows():
        eje.arrow(0, 0, fila["PC1"] * escala, fila["PC2"] * escala, color="gray", alpha=0.6,
                  head_width=0.08, length_includes_head=True)
        eje.text(fila["PC1"] * escala * 1.08, fila["PC2"] * escala * 1.08, var_nombre, fontsize=9,
                 color="dimgray", ha="center", va="center")
    eje.axhline(0, color="lightgray", linewidth=0.8)
    eje.axvline(0, color="lightgray", linewidth=0.8)
    eje.set_xlabel(f"PC1 ({var[0]:.1f}% de la varianza)")
    eje.set_ylabel(f"PC2 ({var[1]:.1f}% de la varianza)")
    eje.set_title(titulo)
    eje.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Nombres de los tipos de vino
# ---------------------------------------------------------------------------
def identificar_tipos(perfiles: pd.DataFrame) -> dict[int, str]:
    """Asigna un arquetipo a cada cluster a partir de su perfil z (regla explícita, no hardcodeada por ID).

    - robusto: máximo de z(Prolina) + z(Alcohol) + z(Flavonoides).
    - intenso: entre los restantes, máximo de z(Ácido málico) + z(Intensidad del color) − z(Tono) − z(Flavonoides).
    - ligero: el cluster restante.
    Pensado para k = 3; con otro k lanza ``ValueError``.
    """
    if len(perfiles) != 3:
        raise ValueError("La regla de nombres está definida para k = 3 clusters.")
    p = perfiles
    robusto = (p["Prolina"] + p["Alcohol"] + p["Flavonoides"]).idxmax()
    resto = p.drop(index=robusto)
    intenso = (resto["Ácido málico"] + resto["Intensidad del color"] - resto["Tono"] - resto["Flavonoides"]).idxmax()
    ligero = resto.drop(index=intenso).index[0]
    return {int(robusto): "robusto", int(intenso): "intenso", int(ligero): "ligero"}


NOMBRES_ARQUETIPO = {
    "robusto": "Tintos robustos y estructurados de gama alta",
    "intenso": "Tintos intensos y ácidos, de perfil evolucionado",
    "ligero": "Tintos ligeros y jóvenes, de color claro",
}


def _fmt(valor: float, dec: int = 2) -> str:
    """Formatea un número con coma decimal (convención en español)."""
    return f"{valor:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fichas_tipos(centroides: pd.DataFrame, perfiles: pd.DataFrame, media_global: pd.Series,
                 arquetipos: dict[int, str]) -> dict[int, dict]:
    """Ficha de cada tipo de vino: nombre, perfil químico (con valores), perfil sensorial y uso comercial.

    Los valores provienen de los centroides des-escalados; el perfil sensorial y la
    sugerencia comercial son INFERENCIAS a partir de la química, no mediciones.
    """
    fichas = {}
    for c, arq in arquetipos.items():
        v, z, g = centroides.loc[c], perfiles.loc[c], media_global

        def dato(var, dec=2):
            return f"{var} {_fmt(v[var], dec)} (media global {_fmt(g[var], dec)}; z = {_fmt(z[var])})"

        if arq == "robusto":
            quimica = [dato("Prolina", 0), dato("Alcohol"), dato("Flavonoides"), dato("Fenoles totales"),
                       dato("OD280/OD315")]
            descripcion = (
                f"Es el grupo de mayor graduación alcohólica ({_fmt(v['Alcohol'])}%) y con la Prolina más alta "
                f"({_fmt(v['Prolina'], 0)} frente a {_fmt(g['Prolina'], 0)} de media), marcadores de uvas muy maduras. "
                f"Concentra la mayor carga de polifenoles nobles: Flavonoides {_fmt(v['Flavonoides'])}, Fenoles totales "
                f"{_fmt(v['Fenoles totales'])} y Proantocianinas {_fmt(v['Proantocianinas'])}, con la menor proporción "
                f"de Fenoles no flavonoides ({_fmt(v['Fenoles no flavonoides'])}). El OD280/OD315 alto "
                f"({_fmt(v['OD280/OD315'])}) y un Tono alto ({_fmt(v['Tono'])}, matices rojos vivos) completan un perfil de "
                "vino con cuerpo, taninos de calidad y aptitud para la guarda."
            )
            sensorial = ("Cuerpo pleno, alcohol perceptible, taninos abundantes pero finos, color estable; "
                         "potencial de envejecimiento en barrica y botella.")
            comercial = ("Segmento premium / reserva. Maridaje con carnes rojas asadas, caza, guisos y quesos "
                         "duros estacionados.")
        elif arq == "intenso":
            quimica = [dato("Intensidad del color"), dato("Tono"), dato("Ácido málico"), dato("Flavonoides"),
                       dato("OD280/OD315")]
            descripcion = (
                f"Tiene la mayor Intensidad del color ({_fmt(v['Intensidad del color'])}) pero el Tono más bajo "
                f"({_fmt(v['Tono'])}), es decir, color profundo con matices teja/anaranjados propios de un vino "
                f"evolucionado. Es el más ácido: Ácido málico {_fmt(v['Ácido málico'])} (media {_fmt(g['Ácido málico'])}). "
                f"Sus Flavonoides son los más bajos ({_fmt(v['Flavonoides'])}) y los Fenoles no flavonoides los más "
                f"altos ({_fmt(v['Fenoles no flavonoides'])}), con OD280/OD315 mínimo ({_fmt(v['OD280/OD315'])}): "
                "la fracción fenólica es menos 'noble', lo que sugiere taninos más rústicos pese al color intenso."
            )
            sensorial = ("Color oscuro con reflejos teja, acidez marcada y vibrante, taninos menos pulidos; "
                         "vino de consumo más temprano que de larga guarda.")
            comercial = ("Segmento medio. Acompaña bien platos grasos o con tomate (pastas, pizzas, embutidos) "
                         "donde la acidez limpia el paladar.")
        else:
            quimica = [dato("Alcohol"), dato("Intensidad del color"), dato("Prolina", 0), dato("Tono"),
                       dato("Magnesio", 1)]
            descripcion = (
                f"Es el grupo de menor graduación ({_fmt(v['Alcohol'])}%) y menor Intensidad del color "
                f"({_fmt(v['Intensidad del color'])}, frente a {_fmt(g['Intensidad del color'])} de media), con la "
                f"Prolina más baja ({_fmt(v['Prolina'], 0)}): uvas menos concentradas. El Tono alto ({_fmt(v['Tono'])}) "
                f"indica matices rojo-púrpura de vino joven. Sus polifenoles están en valores medios (Flavonoides "
                f"{_fmt(v['Flavonoides'])}) y su acidez málica es baja ({_fmt(v['Ácido málico'])}); también presenta "
                f"el menor contenido mineral (Ceniza {_fmt(v['Ceniza'])}, Magnesio {_fmt(v['Magnesio'], 1)})."
            )
            sensorial = ("Cuerpo ligero, color claro y vivo, alcohol moderado, taninos presentes pero sin "
                         "gran concentración; perfil fresco y fácil de beber.")
            comercial = ("Segmento de entrada / consumo cotidiano, vino por copa. Maridaje con aperitivos, "
                         "carnes blancas, pescados grasos o servido ligeramente fresco.")
        fichas[c] = {"nombre": NOMBRES_ARQUETIPO[arq], "arquetipo": arq, "perfil químico": quimica,
                     "descripción": descripcion, "perfil sensorial (inferido)": sensorial,
                     "uso comercial (inferido)": comercial}
    return dict(sorted(fichas.items()))


def fichas_a_markdown(fichas: dict[int, dict], tamanos: pd.Series) -> str:
    """Convierte las fichas de los tipos de vino a texto Markdown."""
    partes = []
    for c, f in fichas.items():
        partes.append(f"### C{c} · {f['nombre']} ({tamanos[c]} vinos)\n")
        partes.append(f"{f['descripción']}\n")
        partes.append("**Perfil químico dominante:**\n")
        partes.extend(f"- {q}" for q in f["perfil químico"])
        partes.append(f"\n**Perfil sensorial (inferencia):** {f['perfil sensorial (inferido)']}\n")
        partes.append(f"**Uso comercial sugerido (inferencia):** {f['uso comercial (inferido)']}\n")
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Validación post-hoc (etiquetas ocultas)
# ---------------------------------------------------------------------------
def validacion_posthoc(etiquetas: np.ndarray, variedad_real: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Contingencia cluster × variedad real, ARI, NMI y pureza.

    La pureza es la proporción de vinos que pertenecen a la variedad mayoritaria de su cluster.
    """
    real = np.asarray(variedad_real)
    contingencia = pd.crosstab(pd.Series(etiquetas, name="Cluster"), pd.Series(real, name="Variedad real"))
    metricas = {
        "ARI": adjusted_rand_score(real, etiquetas),
        "NMI": normalized_mutual_info_score(real, etiquetas),
        "Pureza": float(contingencia.max(axis=1).sum() / contingencia.to_numpy().sum()),
        "Vinos fuera de la variedad mayoritaria": int(contingencia.to_numpy().sum() - contingencia.max(axis=1).sum()),
    }
    return contingencia, metricas


def graficar_pca_clusters_vs_real(proyecciones: pd.DataFrame, etiquetas: np.ndarray,
                                  variedad_real: pd.Series, pca: PCA) -> plt.Figure:
    """PCA lado a lado: coloreado por cluster obtenido y por variedad real (solo post-hoc)."""
    var = pca.explained_variance_ratio_ * 100
    real = np.asarray(variedad_real)
    # Cada variedad se pinta con el color del cluster con el que más coincide, para comparar a simple vista.
    correspondencia = pd.crosstab(real, etiquetas).idxmax(axis=1).to_dict()
    fig, ejes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for eje, grupos, titulo, prefijo in [(ejes[0], etiquetas, "Clusters obtenidos (K-Means)", "Cluster"),
                                         (ejes[1], real, "Variedad real (etiqueta oculta)", "Variedad")]:
        for g in sorted(np.unique(grupos)):
            m = grupos == g
            c = g if prefijo == "Cluster" else correspondencia[g]
            sufijo = "" if prefijo == "Cluster" else f", ≈ Cluster {c}"
            eje.scatter(proyecciones.loc[m, "PC1"], proyecciones.loc[m, "PC2"], s=40, alpha=0.75,
                        color=PALETA.get(c), label=f"{prefijo} {g} (n={m.sum()}{sufijo})")
        errores = None
        if prefijo == "Variedad":
            errores = etiquetas != _alinear(etiquetas, real)
        if errores is not None and errores.any():
            eje.scatter(proyecciones.loc[errores, "PC1"], proyecciones.loc[errores, "PC2"], s=160,
                        facecolors="none", edgecolors="black", linewidths=1.5,
                        label=f"Asignación distinta a la variedad ({errores.sum()})")
        eje.set_title(titulo)
        eje.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        eje.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        eje.legend(fontsize=9)
    fig.suptitle("Validación post-hoc: clusters vs. variedad real en el plano PCA", fontsize=14)
    fig.tight_layout()
    return fig


def _alinear(etiquetas: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Traduce cada variedad real al cluster con el que más coincide (para marcar discrepancias)."""
    contingencia = pd.crosstab(real, etiquetas)
    mapeo = contingencia.idxmax(axis=1).to_dict()
    return np.vectorize(mapeo.get)(real)
