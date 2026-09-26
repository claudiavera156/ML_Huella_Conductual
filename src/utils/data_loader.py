"""Carga de los exports Tractive de los cuatro perros y paso a tablas.

Lee los JSON crudos de data/<perro>/ y genera los tres datasets del proyecto.
Nunca lee position_reports, geofences ni pet_zones, así que ningún dataset
derivado contiene coordenadas.

Las decisiones de lectura están justificadas en
src/notebooks/00_comprension_datos.ipynb.
"""

import json

import pandas as pd

# Categorías de actividad del tracker (00 §4)
CATEGORIAS = {6: "reposo", 0: "calma", 1: "activo", 7: "intenso"}
COLUMNAS_MINUTOS = ["min_reposo", "min_calma", "min_activo", "min_intenso",
                    "min_sin_datos", "min_otros"]

# Un tramo continuo de categoría "intensa" de más de 2 horas no es actividad
# real sino un hueco de datos (00_comprension_datos, §4.1): se cuenta como tiempo sin datos.
MAX_TRAMO_INTENSO_SEG = 2 * 60 * 60


def leer_json(carpeta_datos, perro, fichero):
    with open(f"{carpeta_datos}/{perro}/{fichero}.json") as f:
        return json.load(f)


def fecha_local(dia):
    # gmtTime viene en milisegundos UTC; gmtOffset lo pasa a la hora local
    return pd.to_datetime(dia["gmtTime"] + dia["gmtOffset"], unit="ms").date()


def columna_del_tramo(duracion, categoria):
    if categoria == 7 and duracion > MAX_TRAMO_INTENSO_SEG:
        return "min_sin_datos"
    if categoria in CATEGORIAS:
        return "min_" + CATEGORIAS[categoria]
    if categoria == -1 or categoria is None:
        return "min_sin_datos"
    return "min_otros"


def actividad_diaria(carpeta_datos, perro):
    """Minutos de cada categoría de actividad por día."""
    filas = []
    for dia in leer_json(carpeta_datos, perro, "activity_data"):
        fila = {"perro": perro, "fecha": fecha_local(dia)}
        for columna in COLUMNAS_MINUTOS:
            fila[columna] = 0.0
        for duracion, categoria in dia["activityCategories"]:
            fila[columna_del_tramo(duracion, categoria)] += duracion / 60
        filas.append(fila)
    return pd.DataFrame(filas)


def actividad_horaria(carpeta_datos, perro):
    """Minutos de cada categoría de actividad por hora del día."""
    filas = []
    for dia in leer_json(carpeta_datos, perro, "activity_data"):
        fecha = fecha_local(dia)
        horas = []
        for hora in range(24):
            fila = {"perro": perro, "fecha": fecha, "hora": hora}
            for columna in COLUMNAS_MINUTOS:
                fila[columna] = 0.0
            horas.append(fila)

        inicio = 0  # segundo del día en que empieza el tramo
        for duracion, categoria in dia["activityCategories"]:
            columna = columna_del_tramo(duracion, categoria)
            fin = inicio + duracion
            # un tramo puede ocupar varias horas: se reparte entre ellas
            for hora in range(inicio // 3600, min(fin // 3600 + 1, 24)):
                solape = min(fin, (hora + 1) * 3600) - max(inicio, hora * 3600)
                if solape > 0:
                    horas[hora][columna] += solape / 60
            inicio = fin
        filas = filas + horas
    return pd.DataFrame(filas)


def ladridos_diarios(carpeta_datos, perro):
    """Total de ladridos del día, franjas con ladridos y máximo en una franja."""
    filas = []
    for dia in leer_json(carpeta_datos, perro, "barks"):
        valores = [v for v in dia["periods"] if v]
        filas.append({
            "perro": perro,
            "fecha": pd.to_datetime(dia["local_date"]).date(),
            "ladridos_total": sum(valores),
            "ladridos_franjas": len(valores),
            "ladridos_max_30min": max(valores) if valores else 0,
        })
    return pd.DataFrame(filas)


def ladridos_por_franja(carpeta_datos, perro):
    """Una fila por día y franja de 30 minutos."""
    filas = []
    for dia in leer_json(carpeta_datos, perro, "barks"):
        fecha = pd.to_datetime(dia["local_date"]).date()
        for franja, valor in enumerate(dia["periods"]):
            filas.append({"perro": perro, "fecha": fecha,
                          "franja_30min": franja, "ladridos": valor or 0})
    return pd.DataFrame(filas)


def vitales_diarios(carpeta_datos, perro, fichero, prefijo):
    """Media, mínimo, máximo y nº de muestras de una constante vital por día."""
    filas = []
    for dia in leer_json(carpeta_datos, perro, fichero):
        muestras = [s for registro in dia["records"] for s in registro["samples"]]
        if len(muestras) == 0:
            continue
        serie = pd.Series(muestras)
        filas.append({
            "perro": perro,
            "fecha": pd.to_datetime(dia["local_date"]).date(),
            f"{prefijo}_media": serie.mean(),
            f"{prefijo}_min": serie.min(),
            f"{prefijo}_max": serie.max(),
            f"{prefijo}_n": len(serie),
        })
    return pd.DataFrame(filas)


def temperatura_diaria(carpeta_datos, perro):
    """Temperatura del dispositivo por día, como aproximación a la ambiental."""
    lecturas = []
    for informe in leer_json(carpeta_datos, perro, "hardware_reports"):
        if informe.get("temperature") is not None:
            lecturas.append({"fecha": pd.to_datetime(informe["time"]).date(),
                             "temp": informe["temperature"]})
    por_dia = pd.DataFrame(lecturas).groupby("fecha")["temp"]
    resultado = pd.DataFrame({"temp_disp_media": por_dia.mean(),
                              "temp_disp_max": por_dia.max()}).reset_index()
    resultado["perro"] = perro
    return resultado


def dataset_diario(carpeta_datos, perros):
    """Una fila por perro y día con todas las señales juntas."""
    partes = []
    for perro in perros:
        df = actividad_diaria(carpeta_datos, perro)
        otras_senales = [
            ladridos_diarios(carpeta_datos, perro),
            vitales_diarios(carpeta_datos, perro, "resting_heart_rates", "fc"),
            vitales_diarios(carpeta_datos, perro, "resting_respiratory_rates", "fr"),
            temperatura_diaria(carpeta_datos, perro),
        ]
        for otra in otras_senales:
            if len(otra) == 0:  # p. ej. kiwi no tiene sensor de ladridos: quedan NaN
                continue
            df = df.merge(otra, on=["perro", "fecha"], how="left")
        partes.append(df)

    diario = pd.concat(partes, ignore_index=True)
    diario["fecha"] = pd.to_datetime(diario["fecha"])
    diario["dia_semana"] = diario["fecha"].dt.day_name()
    diario["es_finde"] = diario["fecha"].dt.dayofweek >= 5
    diario["mes"] = diario["fecha"].dt.month
    return diario.sort_values(["perro", "fecha"]).reset_index(drop=True)


def guardar_datasets(carpeta_datos, perros, carpeta_destino):
    """Genera los tres CSV del proyecto y devuelve sus dimensiones."""
    diario = dataset_diario(carpeta_datos, perros)
    franjas = pd.concat([ladridos_por_franja(carpeta_datos, p) for p in perros],
                        ignore_index=True)
    horaria = pd.concat([actividad_horaria(carpeta_datos, p) for p in perros],
                        ignore_index=True)
    franjas = franjas.sort_values(["perro", "fecha", "franja_30min"])
    horaria = horaria.sort_values(["perro", "fecha", "hora"])

    diario.to_csv(f"{carpeta_destino}/diario_perros.csv", index=False)
    franjas.to_csv(f"{carpeta_destino}/ladridos_30min.csv", index=False)
    horaria.to_csv(f"{carpeta_destino}/actividad_horaria.csv", index=False)
    return {"diario_perros": diario.shape, "ladridos_30min": franjas.shape,
            "actividad_horaria": horaria.shape}
