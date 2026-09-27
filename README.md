# Clustering de vinos — Aprendizaje Artificial (Maestría en IA, CAECE)

Trabajo práctico de **aprendizaje no supervisado** de la Unidad 2. Se agrupan los 178 vinos tintos del *Wine Data Set* de UCI a partir de sus 13 variables químicas, sin usar la variedad real, y se nombran los tipos de vino encontrados.

- **Entregable principal:** [`notebooks/clustering_vinos.ipynb`](notebooks/clustering_vinos.ipynb). Narrativa, código y gráficos siguiendo CRISP-DM.
- **Informe académico:** [`informe.md`](informe.md).
- **Resultado:** 3 tipos de vino (k = 3 por consenso de 5 criterios) con K-Means sobre datos estandarizados. Silueta 0,285; estabilidad entre semillas ARI ≥ 0,98; validación post-hoc ARI 0,897 y pureza 96,6 %.

## Estructura

```
├── data/
│   ├── raw/            # etiquetas_ocultas.csv (variedad real, solo para validación post-hoc)
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
├── informe.md
└── requirements.txt
```

**Datos:** si hay un CSV provisto por la cátedra en `data/raw/`, se usa ese archivo. Se aceptan nombres de columna en inglés o en español, con o sin mayúsculas y acentos. Si no hay CSV, se usa `sklearn.datasets.load_wine()`. En ambos casos la variedad real se descarta del análisis y se guarda aparte.

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
