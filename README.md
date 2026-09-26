# ML — La Huella Conductual

Proyecto de Machine Learning con los datos de los collares Tractive de tres perros que conviven: Cherry, Coco y Plum. A partir de un día de datos del collar, ¿se puede reconocer de qué perro se trata? El proyecto recorre el proceso completo: comprensión de los datos, EDA, comparativa de modelos, optimización y evaluación final.

## Problema

Los dueños y el veterinario quieren detectar a tiempo cambios en el comportamiento de cada perro. Los avisos personalizados solo tienen sentido si los datos del collar son lo bastante propios de cada perro como para darle su propia referencia. El modelo responde a esa pregunta: a partir de un día de datos del collar, ¿se puede reconocer de qué perro se trata?

- **Tipo de problema:** clasificación supervisada con tres clases (`perro`).
- **Alcance:** cherry, coco y plum, que llevan el mismo modelo de collar. Kiwi, el cuarto perro de la casa, queda fuera porque su collar es de otro modelo y el modelo podría aprender a reconocer el dispositivo en lugar del perro.
- **Decisión que apoya:** si el modelo reconoce bien a cada perro, los avisos se basan en la referencia de cada uno; si no, basta con un umbral común.

## Dataset

- **Origen:** export oficial de los datos de los collares Tractive (datos privados del autor).
- **Privacidad:** los exports originales **no se publican**, porque algunos ficheros contienen coordenadas GPS de la casa. El repositorio incluye solo datasets derivados, que no contienen ninguna localización. `src/utils/data_loader.py` los genera y nunca lee los ficheros con coordenadas.

| Dataset (`src/data_sample/`) | Una fila por | Contenido |
|---|---|---|
| `diario_perros.csv` | perro y día (898 filas) | minutos por categoría de actividad, ladridos, frecuencia cardíaca y respiratoria en reposo, temperatura del collar y calendario |
| `ladridos_30min.csv` | perro, día y franja de 30 min (27.312 filas) | ladridos por franja |
| `actividad_horaria.csv` | perro, día y hora (21.552 filas) | minutos por categoría de actividad en cada hora |

Los datasets incluyen a los cuatro perros; `main.ipynb` se queda con los tres del alcance. `src/notebooks/00_comprension_datos.ipynb` explica qué contienen los JSON exportados, cómo se pasan a tablas, e incluye el diccionario de datos.

## Solución adoptada

1. **Limpieza:** días con al menos 720 minutos medidos, actividad como porcentaje del tiempo medido y ventana con datos de los tres perros: 495 días.
2. **División temporal:** train hasta el 31 de julio (377 días) y test en agosto y septiembre (118 días). Todo el EDA y las decisiones del modelo se hacen solo con train.
3. **EDA:** univariante, cuatro hipótesis contrastadas (ANOVA, U de Mann-Whitney, correlación de Pearson y chi-cuadrado) y multivariante.
4. **Features:** nueve señales de actividad, constantes vitales en reposo y ladridos, elegidas a partir del EDA.
5. **Modelado:** baseline y comparativa de siete modelos con validación cruzada sobre train y F1 macro como métrica.
6. **Optimización:** regresión logística y XGBoost. Quedan empatados y se elige la regresión logística, porque sus coeficientes muestran qué señales distinguen a cada perro.

## Principales resultados

**EDA**

| Hipótesis | Veredicto |
|---|---|
| H1 — La frecuencia cardíaca y respiratoria en reposo difieren entre los perros | **Confirmada en parte**: la FR separa a los tres; la FC no separa a coco de plum |
| H2 — En los días más calurosos los perros están menos activos | **Confirmada con matices**: menos actividad y FR más alta en los meses cálidos, pero no día a día dentro del verano |
| H3 — Los perros que conviven ladran a la vez más de lo esperable por azar | **Confirmada**: coinciden de 1,41 a 1,85 veces más de lo esperable incluso en horas punta |
| H4 — Los perros están más activos en fin de semana | **Refutada**: están ligeramente menos activos, con la diferencia entre las 9:00 y las 12:00 |

- Cada perro tiene una huella propia en el **nivel** de sus señales (plum apenas ladra, coco pasa menos tiempo en calma), pero el **horario de actividad es común** a los tres.
- Los datos del collar contienen un **artefacto**: bloques continuos de hasta 13 horas de actividad "intensa" que no son reales. El loader los corrige.

**Modelo**

- **En test, con 118 días que el modelo no había visto: F1 macro de 0,983 y accuracy de 0,983.** El baseline, que responde siempre el perro más frecuente, se queda en un F1 macro de 0,168 en validación cruzada.
- **Solo hay 2 errores:** dos días seguidos de plum, a principios de septiembre, reconocidos como cherry, en días en que su frecuencia respiratoria subió hasta parecerse a la de cherry.
- **Las señales que más distinguen a cada perro:** los pocos ladridos de plum; el menor tiempo en calma y la mayor actividad de coco; y, en cherry, más tiempo en calma, una frecuencia cardíaca más baja y ladridos repartidos en más franjas del día.
- **Conclusión:** los datos del collar son lo bastante propios de cada perro como para basar los avisos en la referencia individual de cada uno.

**Limitaciones principales:** son tres perros de una misma casa, cada perro ha llevado siempre el mismo collar y el modelo no ha visto datos de otoño ni de invierno.

## Estructura del repositorio

```
├── main.ipynb                          # EDA, modelado y evaluación
├── Presentacion.pdf                    # documento soporte de la presentación
├── README.md
├── requirements.txt
├── src/
│   ├── data_sample/                    # datasets derivados (sin coordenadas)
│   ├── img/                            # gráficos del proyecto
│   ├── models/                         # modelo final y escalador (pickle)
│   ├── notebooks/
│   │   └── 00_comprension_datos.ipynb  # JSON crudos → tablas, y diccionario de datos
│   └── utils/
│       ├── data_loader.py              # generación de los datasets a partir de los exports
│       └── bootcampviztools.py         # funciones de visualización
```

## Reproducción

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook main.ipynb   # desde la raíz del repositorio, o abrir en VS Code con el kernel .venv
```

`main.ipynb` solo necesita los datasets de `src/data_sample/`. Para regenerarlos hacen falta los exports originales en una carpeta `data/<perro>/` situada junto al repositorio; después se ejecuta `src/notebooks/00_comprension_datos.ipynb`.

## Tecnologías

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn

## Autora

Claudia Vera de Paul
