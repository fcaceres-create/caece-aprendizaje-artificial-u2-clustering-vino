# Clustering de vinos — Aprendizaje Artificial (Maestría en IA, CAECE)

| | |
|---|---|
| **Universidad** | Universidad CAECE |
| **Materia** | Aprendizaje Artificial — Maestría en Inteligencia Artificial |
| **Unidad** | 2: Clasificación, Clustering y Reglas de Asociación |
| **Alumno** | Fernando Cáceres |
| **Profesores** | Juan Azcurra y Pablo Hernán Paul |
| **Fecha** | 09/2026 |

Trabajo práctico de **aprendizaje no supervisado** de la Unidad 2. Se agrupan los 178 vinos tintos del *Wine Data Set* de UCI a partir de sus 13 variables químicas, sin usar la variedad real, y se nombran los tipos de vino encontrados.

- **Entregable principal:** [`notebooks/clustering_vinos.ipynb`](notebooks/clustering_vinos.ipynb). Narrativa, código y gráficos siguiendo CRISP-DM.
- **Informe académico:** [`informe.md`](informe.md).
- **Resultado:** 3 tipos de vino (k = 3 por consenso de 5 criterios) con K-Means sobre datos estandarizados. Silueta 0,285; estabilidad entre semillas ARI ≥ 0,98; validación post-hoc ARI 0,897 y pureza 96,6 %.

## Cómo verlo sin instalar nada

- **Notebook con salidas y gráficos (recomendado):** [abrir en nbviewer](https://nbviewer.org/github/fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino/blob/main/notebooks/clustering_vinos.ipynb).
- **En GitHub:** el [notebook](notebooks/clustering_vinos.ipynb) y el [informe](informe.md) se muestran directamente en el repositorio, con las figuras incluidas.
- **Para ejecutarlo en Google Colab:** [abrir en Colab](https://colab.research.google.com/github/fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino/blob/main/notebooks/clustering_vinos.ipynb). Para volver a correrlo, el notebook necesita el paquete `src/`, así que antes hay que ejecutar en una celda nueva al principio:
  ```python
  !git clone https://github.com/fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino.git
  %cd caece-aprendizaje-artificial-u2-clustering-vino/notebooks
  ```

## Laboratorio web interactivo (Streamlit)

[`app.py`](app.py) es una web con solapas (Resumen, Datos y EDA, Preparación, Elección de k, Modelo y estabilidad, Comparar algoritmos, Tipos de vino y Validación post-hoc) para experimentar con el trabajo:

- **Datos:** usar el CSV de la cátedra, el dataset de UCI o subir un CSV propio; editar valores en la tabla; elegir qué variables se usan para clusterizar.
- **Parámetros:** escalado, tratamiento de outliers, k (automático por consenso o manual), algoritmo (K-Means o jerárquico con distintos linkage y distancias), `init`, `n_init`, `max_iter` y semilla.
- **Informe recreado:** todos los textos se generan a partir de los resultados. Se puede descargar un informe en Markdown con la configuración actual y el CSV de vinos con su cluster.
- Botón para abrir el **notebook original en nbviewer**.

Para correrla en tu equipo (con el entorno ya instalado):

```bash
streamlit run app.py
```

Se abre en http://localhost:8501.

**Publicarla gratis en Streamlit Community Cloud:**

1. Entrar a https://share.streamlit.io con la cuenta de GitHub.
2. *Create app* → *Deploy a public app from GitHub*. Repositorio: `fcaceres-create/caece-aprendizaje-artificial-u2-clustering-vino`, rama `main`, archivo `app.py`.
3. En *Advanced settings*, elegir la versión de Python más nueva disponible.
4. *Deploy*. Cada push a `main` actualiza la app automáticamente.

## Estructura

```
├── data/
│   ├── raw/            # wine-clustering.csv (CSV de la cátedra) + etiquetas_ocultas.csv (solo post-hoc)
│   └── processed/      # vinos_escalado.csv (StandardScaler)
├── notebooks/
│   └── clustering_vinos.ipynb   # ENTREGABLE PRINCIPAL
├── src/
│   ├── data.py           # carga, renombrado al español, E/S de figuras y tablas
│   ├── eda.py            # análisis exploratorio y gráficos
│   ├── preprocessing.py  # escalado y alternativa de tratamiento de outliers
│   ├── clustering.py     # codo, silueta, dendrograma, K-Means, estabilidad, jerárquico
│   ├── interpretation.py # perfiles, PCA, nombres de los tipos, validación post-hoc
│   └── run_all.py        # pipeline completo
├── outputs/
│   ├── figures/        # PNG a 150 dpi
│   └── tables/         # CSV + Markdown
├── app.py            # laboratorio web interactivo (Streamlit)
├── .streamlit/       # tema visual de la app
├── informe.md
└── requirements.txt
```

**Datos:** se usa el CSV provisto por la cátedra, `data/raw/wine-clustering.csv`. Se aceptan nombres de columna en inglés o en español, con o sin mayúsculas y acentos. Si no hubiera CSV, se usaría `sklearn.datasets.load_wine()`. Como el CSV coincide fila a fila con UCI, la variedad real se toma de scikit-learn y se guarda aparte, solo para la validación post-hoc.

## Instalación y ejecución

Requiere Python 3.11 o superior (probado con Python 3.14).

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Regenerar todas las figuras y tablas
python -m src.run_all

# Ejecutar el notebook de arriba a abajo
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering_vinos.ipynb
# o abrirlo de forma interactiva
jupyter notebook notebooks/clustering_vinos.ipynb
```

Si PowerShell bloquea la activación del entorno, ejecutar antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_all
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering_vinos.ipynb
```

`python -m src.run_all` imprime un resumen: la elección de k, los tamaños, la estabilidad, los tipos de vino y la validación post-hoc. También informa cuántos warnings se registraron durante la ejecución (0 en la versión entregada).

## Reproducibilidad

Todo lo estocástico usa `random_state=42` y además se fija `np.random.seed(42)`. Los clusters se renumeran por Prolina media descendente, de modo que los IDs y los nombres no dependen de la inicialización aleatoria.
