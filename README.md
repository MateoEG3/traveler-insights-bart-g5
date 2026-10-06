# Traveler Insights UPTC 2026 — Grupo 5 (BART)

Proyecto final de NLP. Clasificación de reseñas turísticas de Sucre (Foursquare y Tripadvisor) en
`negativo`, `neutral` y `positivo`, con un modelo BART ajustado por el grupo y documentado con CRISP-ML(Q).

- **Competencia:** https://www.kaggle.com/competitions/traveler-insights-uptc-2026
- **Notebook de Kaggle (privado, compartido con el docente):** [pegar enlace]
- **Perfil del equipo en Kaggle:** [pegar enlace]

## Integrantes

| Integrante | GitHub | Kaggle |
|---|---|---|
| Mateo Echeverría González | @[usuario] | [usuario] |
| Juan David López | @[usuario] | [usuario] |

Los roles (datos, modelado, evaluación, documentación) rotan; el registro está en
[`docs/bitacora_trabajo.md`](docs/bitacora_trabajo.md).

## Estructura

```
notebooks/team-bart.ipynb     notebook de Kaggle ejecutado (se actualiza con cada versión)
resultados/experimentos.csv   registro de experimentos (lo genera el notebook)
resultados/folds.csv          partición usada (la genera el notebook)
docs/bitacora_trabajo.md      qué se hizo, quién y con qué rol
docs/bitacora_ia.md           uso de IA generativa (anexo del documento)
docs/                         documento final en PDF (al cierre)
requirements.txt              versiones del entorno donde se obtuvieron los resultados
```

Los datos de la competencia y los pesos del modelo no se versionan (ver "Cómo reproducir").

## Cómo reproducir

1. Entrar a la competencia en Kaggle y aceptar las reglas.
2. Crear un notebook desde la competencia e importar `notebooks/team-bart.ipynb` (File → Import Notebook).
3. En la configuración de la sesión: Internet activado. GPU a partir de la parte de BART.
4. Ejecutar todo (Run All). El notebook localiza `train.csv`, `test.csv` y `submission.csv` en `/kaggle/input`.
5. Las salidas quedan en `/kaggle/working`: `folds.csv`, `experimentos.csv` y el archivo de envío.

Semilla global: 42. Las cifras de referencia son las obtenidas en Kaggle con las versiones de `requirements.txt`.

## Estado

| Parte | Fases CRISP-ML(Q) | Estado |
|---|---|---|
| 1 | Datos (1), ingeniería de datos (2), baseline (3) | Hecha |
| 2 | Fine-tuning de BART (3) | Pendiente |
| 3 | Evaluación (4), despliegue (5), monitoreo (6) | Pendiente |

## Resultados hasta ahora (hold-out local de 351 reseñas)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Clase mayoritaria | 0,781 | 0,292 |
| TF-IDF + Regresión Logística, texto | 0,863 | 0,518 |
| TF-IDF + Regresión Logística, texto, pesos de clase | 0,900 | 0,721 |
| TF-IDF + Regresión Logística, texto + metadatos del lugar | 0,997 | 0,997 |

Hallazgo principal de la Fase 1: la etiqueta `Sentimiento` se deriva de `Valoración_num` (calificación
promedio del lugar) mediante umbrales, y no del texto de la reseña. Detalle y evidencia en el notebook.
