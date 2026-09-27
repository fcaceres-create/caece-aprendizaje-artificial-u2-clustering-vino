"""Ejecuta el pipeline completo y regenera todas las figuras y tablas.

Uso:  python -m src.run_all
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # sin ventanas: solo se guardan archivos

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import clustering as cl  # noqa: E402
from . import eda  # noqa: E402
from . import interpretation as it  # noqa: E402
from . import preprocessing as pp  # noqa: E402
from .data import (DIR_TABLAS, SEMILLA, cargar_etiquetas_ocultas, cargar_vinos, fijar_semillas,  # noqa: E402
                   guardar_figura, guardar_tabla)

K_RANGO = range(1, 11)


def _figura(fig, nombre: str) -> None:
    """Guarda la figura y libera memoria."""
    guardar_figura(fig, nombre)
    plt.close("all")


def fase_eda(df: pd.DataFrame) -> dict:
    """Fase 2 (CRISP-DM): comprensión de los datos."""
    guardar_tabla(eda.resumen_basico(df), "01_resumen_variables")
    guardar_tabla(eda.contar_outliers(df), "02_outliers")
    corr = eda.matriz_correlacion(df)
    guardar_tabla(eda.correlaciones_fuertes(corr), "03_correlaciones_fuertes", indice=False)
    guardar_tabla(eda.ejemplo_dominancia_escala(df), "04_ejemplo_dominancia_escala")
    guardar_tabla(eda.participacion_varianza(df).to_frame(), "05_participacion_varianza")
    _figura(eda.graficar_histogramas(df), "01_histogramas.png")
    _figura(eda.graficar_boxplots(df), "02_boxplots.png")
    _figura(eda.graficar_boxplots_escala_comun(df), "03_boxplots_escala_comun.png")
    _figura(eda.graficar_correlacion(corr), "04_matriz_correlacion.png")
    _figura(eda.graficar_pairplot(df, eda.seleccionar_variables_pairplot(df)).figure, "05_pairplot.png")
    return {"duplicados": int(df.duplicated().sum()), "nulos": int(df.isna().sum().sum())}


def fase_preparacion(df: pd.DataFrame) -> dict:
    """Fase 3: escalado (StandardScaler) y comparación con alternativas."""
    guardar_tabla(pp.comparar_escaladores(df), "06_comparacion_escaladores")
    df_esc, escalador = pp.escalar(df)
    pp.guardar_procesado(df_esc)
    return {"df_esc": df_esc, "escalador": escalador}


def fase_eleccion_k(X) -> dict:
    """Fase 4.1: codo, silueta, CH, DB y dendrograma."""
    tabla = cl.evaluar_rango_k(X, K_RANGO)
    guardar_tabla(tabla, "07_metricas_por_k")
    k_codo = cl.detectar_codo(tabla["inercia (WCSS)"])
    Z = cl.calcular_linkage(X, "ward")
    saltos = cl.saltos_dendrograma(Z)
    guardar_tabla(saltos, "08_saltos_dendrograma_ward")
    resumen = cl.resumen_eleccion_k(tabla, k_codo, saltos)
    guardar_tabla(resumen, "09_resumen_eleccion_k")
    k = int(resumen["k sugerido"].mode()[0])  # consenso de criterios
    _figura(cl.graficar_codo(tabla, k_codo), "06_metodo_codo.png")
    _figura(cl.graficar_silueta_promedio(tabla, int(tabla["silueta"].idxmax())), "07_silueta_promedio.png")
    _figura(cl.graficar_siluetas_por_cluster(X, [k - 1, k, k + 1]), "08_silueta_por_cluster.png")
    _figura(cl.graficar_dendrograma(Z, saltos.loc[k, "umbral de corte"], k,
                                    "Dendrograma jerárquico (linkage Ward, distancia euclídea, datos escalados)"),
            "09_dendrograma_ward.png")
    Z_man = cl.calcular_linkage(X, "average", "cityblock")
    _figura(cl.graficar_dendrograma(Z_man, cl.saltos_dendrograma(Z_man).loc[k, "umbral de corte"], k,
                                    "Dendrograma de contraste (linkage medio, distancia Manhattan)"),
            "10_dendrograma_manhattan_average.png")
    return {"tabla": tabla, "resumen": resumen, "k": k}


def fase_modelado(df: pd.DataFrame, df_esc: pd.DataFrame, k: int) -> dict:
    """Fases 4.2 a 4.4: K-Means final, estabilidad, sin escalar, outliers y jerárquico."""
    X = df_esc.to_numpy()
    modelo = cl.crear_kmeans(k).fit(X)
    etiquetas_km = modelo.labels_
    estab = pd.DataFrame([cl.estabilidad_semillas(X, k),
                          cl.estabilidad_semillas(X, k, n_init=1, init="random")]).set_index("configuración")
    guardar_tabla(estab, "10_estabilidad_semillas")
    sin_esc, dominancia = cl.comparar_sin_escalar(df, X, etiquetas_km, k)
    guardar_tabla(sin_esc, "12_kmeans_sin_escalar")
    guardar_tabla(dominancia.to_frame(), "13_dominancia_sin_escalar")
    df_wins_esc, _ = pp.escalar(pp.recortar_outliers_iqr(df))
    guardar_tabla(pd.DataFrame([cl.impacto_outliers(df_wins_esc.to_numpy(), etiquetas_km, k)]),
                  "14_impacto_outliers", indice=False)
    jer, et_jer = cl.comparar_jerarquicos(X, k, etiquetas_km)
    guardar_tabla(jer, "15_comparacion_jerarquicos")
    return {"modelo": modelo, "etiquetas_km": etiquetas_km, "estabilidad": estab, "jerarquicos": jer,
            "etiquetas_ward": et_jer["Jerárquico ward (euclidean)"]}


def fase_interpretacion(df, df_esc, escalador, modelo, etiquetas_km, etiquetas_ward) -> dict:
    """Fases 5 y 6: orden determinístico, perfiles, PCA y nombres de los tipos de vino."""
    etiquetas, mapeo = it.ordenar_clusters(etiquetas_km, df)
    et_ward, _ = it.ordenar_clusters(etiquetas_ward, df)
    guardar_tabla(cl.tabla_contingencia(etiquetas, et_ward, "K-Means", "Jerárquico Ward"),
                  "16_contingencia_kmeans_vs_ward")
    tamanos = cl.tamanos_clusters(etiquetas)
    centroides = it.centroides_originales(modelo, escalador, mapeo, df.columns)
    perfiles = it.perfiles_zscore(df_esc, etiquetas)
    arquetipos = it.identificar_tipos(perfiles)
    fichas = it.fichas_tipos(centroides, perfiles, df.mean(), arquetipos)
    nombres = {c: f["nombre"] for c, f in fichas.items()}
    tamanos_nombrados = tamanos.copy()
    tamanos_nombrados.insert(0, "tipo de vino", [nombres[c] for c in tamanos.index])
    guardar_tabla(tamanos_nombrados, "11_tamanos_clusters")
    guardar_tabla(centroides, "17_centroides_unidades_originales")
    guardar_tabla(perfiles, "18_perfiles_zscore")
    guardar_tabla(it.comparar_con_media_global(df, etiquetas), "19_medias_vs_global")
    ranking = it.ranking_discriminantes(df_esc, etiquetas)
    guardar_tabla(ranking, "20_ranking_discriminantes", decimales=4)
    pca, proy, loadings = it.ajustar_pca(df_esc)
    guardar_tabla(loadings, "21_pca_loadings")
    centros_esc = pd.DataFrame(escalador.transform(centroides), columns=df.columns).to_numpy()
    _figura(it.graficar_heatmap_perfiles(perfiles, nombres), "11_heatmap_perfiles.png")
    _figura(it.graficar_boxplots_por_cluster(df, etiquetas, ranking.index[:6].tolist(), nombres),
            "12_boxplots_por_cluster.png")
    _figura(it.graficar_pca(pca, proy, loadings, etiquetas, centros_esc, nombres,
                            "Vinos en el plano PCA (solo visualización), coloreados por cluster"),
            "13_pca_biplot.png")
    (DIR_TABLAS / "tipos_de_vino.md").write_text(it.fichas_a_markdown(fichas, tamanos["vinos"]), encoding="utf-8")
    return {"etiquetas": etiquetas, "tamanos": tamanos, "fichas": fichas, "nombres": nombres,
            "pca": pca, "proy": proy}


def fase_posthoc(etiquetas, pca, proy) -> dict | None:
    """Fase 7: validación con la variedad real (NO interviene en ninguna decisión previa)."""
    real = cargar_etiquetas_ocultas()
    if real is None:
        print("No hay etiquetas ocultas disponibles: se omite la validación post-hoc.")
        return None
    contingencia, metricas = it.validacion_posthoc(etiquetas, real)
    guardar_tabla(contingencia, "22_posthoc_contingencia")
    guardar_tabla(pd.DataFrame([metricas]), "23_posthoc_metricas", indice=False)
    _figura(it.graficar_pca_clusters_vs_real(proy, etiquetas, real, pca), "14_posthoc_pca.png")
    return metricas


def main() -> None:
    """Corre todas las fases e imprime un resumen final."""
    fijar_semillas(SEMILLA)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        df = cargar_vinos()
        info_eda = fase_eda(df)
        prep = fase_preparacion(df)
        eleccion = fase_eleccion_k(prep["df_esc"].to_numpy())
        k = eleccion["k"]
        mod = fase_modelado(df, prep["df_esc"], k)
        interp = fase_interpretacion(df, prep["df_esc"], prep["escalador"], mod["modelo"],
                                     mod["etiquetas_km"], mod["etiquetas_ward"])
        post = fase_posthoc(interp["etiquetas"], interp["pca"], interp["proy"])

    pd.set_option("display.width", 200)
    print(f"Datos: {df.shape[0]} vinos × {df.shape[1]} variables | nulos: {info_eda['nulos']} | "
          f"duplicados: {info_eda['duplicados']}")
    print("\nElección de k:\n", eleccion["resumen"].to_string())
    print(f"\nk final = {k} | silueta = {eleccion['tabla'].loc[k, 'silueta']:.3f}")
    print("\nTamaños:\n", interp["tamanos"].round(1).to_string())
    print("\nEstabilidad (ARI entre 10 semillas):\n", mod["estabilidad"][["ARI medio", "ARI mínimo"]].round(3).to_string())
    print("\nTipos de vino:")
    for c, nombre in interp["nombres"].items():
        print(f"  C{c}: {nombre}")
    if post:
        print("\nValidación post-hoc:", {m: round(v, 3) for m, v in post.items()})
    print(f"\nWarnings registrados: {len(avisos)}")
    for a in avisos:
        print(f"  - {a.category.__name__}: {a.message}")


if __name__ == "__main__":
    main()
