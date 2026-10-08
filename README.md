# Traveler Insights UPTC 2026 — Grupo 5 (BART)

Proyecto final de Procesamiento de Lenguaje Natural (Especialización en Programación para Ciencia de Datos, UPTC).
Clasificación de reseñas turísticas de Sucre (Foursquare y Tripadvisor) en `negativo`, `neutral` y `positivo`,
con un modelo BART ajustado por el grupo y documentado con CRISP-ML(Q).

- **Documento técnico:** [`docs/Documento_tecnico_CRISP-MLQ_Grupo5_BART.pdf`](docs/Documento_tecnico_CRISP-MLQ_Grupo5_BART.pdf)
- **Notebook de Kaggle:** https://www.kaggle.com/code/mateoro3/team-bart/notebook
- **Competencia:** https://www.kaggle.com/competitions/traveler-insights-uptc-2026

## Integrantes

| Integrante | GitHub | Kaggle |
|---|---|---|
| Mateo Echeverría González | [@MateoEG3](https://github.com/MateoEG3) | [mateoro3](https://www.kaggle.com/mateoro3) |
| Juan David López | [@juandavidlopez5](https://github.com/juandavidlopez5) | [juanlopez18](https://www.kaggle.com/juanlopez18) |

Los dos integrantes compartieron los roles de datos, modelado, evaluación y documentación.
Docente: Edwin Alexander Puertas Del Castillo.

## Resultados

| Modelo | Accuracy CV | Accuracy hold-out | F1 macro hold-out | Leaderboard público |
|---|---|---|---|---|
| Clase mayoritaria | 0,778 | 0,781 | 0,292 | – |
| Baseline: TF-IDF + regresión logística, solo texto | 0,903 | 0,900 | 0,754 | 0,87 |
| BART (BARTO), solo texto | 0,886 | 0,863 | 0,606 | – |
| **BART (BARTO), texto + metadatos del lugar (modelo final)** | **1,000** | **0,997** | **0,982** | **1,00** |

CV: validación cruzada de 5 folds sobre 806 reseñas. Hold-out: las 351 reseñas de `test.csv`.

**Hallazgo principal.** La etiqueta `Sentimiento` no proviene del texto de la reseña: es un umbral sobre
`Valoración_num`, la calificación promedio del lugar, y una regla sobre esa columna la reproduce al 100 %.
Por eso el BART que solo lee la reseña no supera al baseline, y el modelo final acierta leyendo la valoración
del lugar, no interpretando la opinión. El detalle y la evidencia están en el documento técnico.

## Estructura

```
docs/Documento_tecnico_CRISP-MLQ_Grupo5_BART.pdf   documento técnico (15 páginas y anexos)
notebooks/team-bart.ipynb                          cuaderno de Kaggle ejecutado, Fases 1 a 6
resultados/experimentos.csv                        registro de experimentos (baselines y 9 corridas de BART)
resultados/experimentos/*.pkl                      predicciones y curvas de cada corrida
resultados/folds.csv                               partición usada (5 folds, semilla 42)
resultados/errores_revisados.csv                   los 48 errores del modelo de solo texto, categorizados
resultados/errores_revisados_con_equipo.csv        revisión manual de esos errores
resultados/simulacion_drift.csv                    simulación de drift
resultados/inferencia.py                           guion que genera el envío desde el modelo guardado
resultados/submission.csv                          envío final a Kaggle
requirements.txt                                   versiones del entorno
```

Los datos de la competencia y los pesos del modelo (567 MB) no se versionan.

## Cómo reproducir

1. Entrar a la competencia en Kaggle y aceptar las reglas.
2. Crear un notebook desde la competencia e importar `notebooks/team-bart.ipynb` (File → Import Notebook).
3. En la configuración de la sesión: Internet activado y acelerador **GPU T4 ×2**.
4. Ejecutar con Save Version → Save & Run All. El cuaderno localiza `train.csv`, `test.csv` y `submission.csv`
   en `/kaggle/input` y deja las salidas en `/kaggle/working`.

Entrenar todo desde cero toma unas 3 horas de GPU. Para reutilizar los experimentos ya entrenados, subir los
archivos de `resultados/experimentos/` como un dataset de Kaggle y agregarlo como Input: el cuaderno los
recupera si la configuración y los folds son idénticos, y la corrida baja a unos 20 minutos.

Para regenerar solo el archivo de envío desde el modelo guardado:

```
python inferencia.py --modelo modelo_final --entrada submission.csv --salida submission.csv
```

Semilla global: 42. Las cifras de referencia son las obtenidas en Kaggle con las versiones de `requirements.txt`.

## Uso de IA generativa

Está declarado en el documento técnico (sección de cierre y Anexo A, bitácora de uso).
