"""Aplicación interactiva de estadística descriptiva.

Este módulo permite cargar una serie de datos numéricos ya sea de forma manual o
mediante un archivo CSV y calcula múltiples medidas estadísticas descriptivas.
Además genera dos gráficos: un histograma y un diagrama de dispersión.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MIN_DATA_POINTS = 10


def _clean_numeric_values(values: Iterable[float]) -> pd.Series:
    """Convierte una colección de valores a una serie numérica.

    Se eliminan valores NaN y se asegura que la longitud mínima se cumpla.
    """
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < MIN_DATA_POINTS:
        raise ValueError(
            f"Se requieren al menos {MIN_DATA_POINTS} valores numéricos (recibidos {len(series)})."
        )
    return series


def cargar_datos_manual() -> pd.Series:
    """Solicita al usuario ingresar los datos manualmente por consola."""
    while True:
        texto = input(
            f"Ingrese al menos {MIN_DATA_POINTS} datos numéricos separados por espacios:\n> "
        ).strip()
        if not texto:
            print("No se recibieron datos. Intente nuevamente.\n")
            continue
        try:
            valores = [float(x.replace(",", ".")) for x in texto.split()]
            return _clean_numeric_values(valores)
        except ValueError as exc:
            print(f"Error al procesar los datos: {exc}. Intente nuevamente.\n")


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un archivo CSV y valida su existencia."""
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo '{ruta}' no existe.")
    try:
        return pd.read_csv(ruta)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc


def cargar_datos_csv() -> pd.Series:
    """Permite seleccionar una columna numérica desde un archivo CSV."""
    while True:
        ruta_txt = input("Ingrese la ruta del archivo CSV:\n> ").strip()
        if not ruta_txt:
            print("Debe proporcionar una ruta válida.\n")
            continue
        ruta = Path(ruta_txt)
        try:
            df = _leer_csv(ruta)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{exc}\n")
            continue

        columnas_numericas = df.select_dtypes(include=["number"]).columns
        if not columnas_numericas.any():
            print(
                "El archivo no contiene columnas numéricas. Asegúrese de que exista al menos una columna numérica.\n"
            )
            continue

        print("Columnas numéricas disponibles:")
        for idx, col in enumerate(columnas_numericas, start=1):
            print(f"  {idx}. {col}")
        seleccion = input("Seleccione el número de la columna a utilizar:\n> ").strip()
        if not seleccion.isdigit() or not (1 <= int(seleccion) <= len(columnas_numericas)):
            print("Selección inválida. Intente nuevamente.\n")
            continue

        columna = columnas_numericas[int(seleccion) - 1]
        try:
            return _clean_numeric_values(df[columna].to_numpy())
        except ValueError as exc:
            print(f"{exc}\n")


def cargar_datos() -> pd.Series:
    """Solicita al usuario el modo de carga de datos."""
    print("=== Ingreso de datos ===")
    while True:
        print("Seleccione una opción:")
        print("  1. Ingresar datos manualmente")
        print("  2. Cargar datos desde un archivo CSV")
        opcion = input("> ").strip()
        if opcion == "1":
            return cargar_datos_manual()
        if opcion == "2":
            return cargar_datos_csv()
        print("Opción inválida. Intente nuevamente.\n")


def calcular_medidas(series: pd.Series) -> dict:
    """Calcula medidas estadísticas solicitadas."""
    media = series.mean()
    mediana = series.median()
    modas = series.mode()
    moda = modas.iloc[0] if not modas.empty else float("nan")

    cuartiles = series.quantile([0.25, 0.5, 0.75])
    deciles = series.quantile([i / 10 for i in range(1, 10)])
    percentiles = series.quantile([i / 100 for i in range(10, 100, 10)])
    percentil_90 = series.quantile(0.9)

    rango = series.max() - series.min()
    varianza = series.var(ddof=1)
    desv_tipica = series.std(ddof=1)
    desv_media = (series - media).abs().mean()
    coef_var = (desv_tipica / media) * 100 if media != 0 else np.nan

    return {
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "cuartiles": cuartiles,
        "deciles": deciles,
        "percentiles": percentiles,
        "percentil_90": percentil_90,
        "rango": rango,
        "varianza": varianza,
        "desv_tipica": desv_tipica,
        "desv_media": desv_media,
        "coef_var": coef_var,
    }


def mostrar_medidas(medidas: dict) -> None:
    """Imprime las medidas calculadas en consola."""
    print("\n=== Resultados estadísticos ===")
    print(
        f"Media: {medidas['media']:.4f}\n"
        f"Mediana: {medidas['mediana']:.4f}\n"
        f"Moda: {medidas['moda']:.4f}"
    )

    print("\nCuartiles:")
    for cuant, valor in medidas["cuartiles"].items():
        print(f"  Q{int(cuant * 4)} ({cuant:.2f}): {valor:.4f}")

    print("\nDeciles:")
    for cuant, valor in medidas["deciles"].items():
        print(f"  D{int(cuant * 10)} ({cuant:.1f}): {valor:.4f}")

    print("\nPercentiles (cada 10%):")
    for cuant, valor in medidas["percentiles"].items():
        print(f"  P{int(cuant * 100)} ({cuant:.2f}): {valor:.4f}")
    print(f"  P90 (0.90): {medidas['percentil_90']:.4f}")

    print("\nMedidas de dispersión:")
    print(
        f"  Rango: {medidas['rango']:.4f}\n"
        f"  Varianza: {medidas['varianza']:.4f}\n"
        f"  Desviación típica: {medidas['desv_tipica']:.4f}\n"
        f"  Desviación media: {medidas['desv_media']:.4f}\n"
        f"  Coeficiente de variación: {medidas['coef_var']:.2f}%"
    )


def generar_graficos(series: pd.Series) -> None:
    """Genera un histograma y un diagrama de dispersión."""
    print("\nGenerando gráficos... cierre la ventana de gráficos para continuar.")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(series, bins="auto", kde=True, color="#3f51b5")
    plt.title("Histograma de frecuencias")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=range(1, len(series) + 1), y=series, color="#009688")
    plt.title("Diagrama de dispersión")
    plt.xlabel("Índice")
    plt.ylabel("Valor")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Punto de entrada de la aplicación."""
    print("=== Aplicación de Estadística Descriptiva ===\n")
    datos = cargar_datos()
    medidas = calcular_medidas(datos)
    mostrar_medidas(medidas)
    generar_graficos(datos)


if __name__ == "__main__":
    main()
