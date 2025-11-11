"""Aplicación interactiva de estadística descriptiva.

Este módulo permite cargar una serie de datos numéricos ya sea de forma manual o
mediante un archivo CSV y calcula múltiples medidas estadísticas descriptivas.
Además genera dos gráficos: un histograma y un diagrama de dispersión.

La aplicación ofrece ahora una interfaz gráfica construida con Tkinter que
simplifica el flujo de carga, cálculo y visualización de resultados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


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


def parsear_datos_manual(texto: str) -> pd.Series:
    """Convierte una cadena con datos numéricos separados por espacios en serie."""

    if not texto.strip():
        raise ValueError("Debe ingresar al menos un valor numérico.")

    try:
        valores = [float(x.replace(",", ".")) for x in texto.split()]
    except ValueError as exc:  # pragma: no cover - errores de conversión en GUI
        raise ValueError("Los datos ingresados contienen valores no numéricos.") from exc
    return _clean_numeric_values(valores)


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
    print(formatear_medidas(medidas))


def formatear_medidas(medidas: dict) -> str:
    """Devuelve una representación textual de las medidas calculadas."""

    lineas = [
        "=== Resultados estadísticos ===",
        (
            f"Media: {medidas['media']:.4f}\n"
            f"Mediana: {medidas['mediana']:.4f}\n"
            f"Moda: {medidas['moda']:.4f}"
        ),
        "",
        "Cuartiles:",
    ]

    lineas.extend(
        f"  Q{int(cuant * 4)} ({cuant:.2f}): {valor:.4f}"
        for cuant, valor in medidas["cuartiles"].items()
    )

    lineas.extend(["", "Deciles:"])
    lineas.extend(
        f"  D{int(cuant * 10)} ({cuant:.1f}): {valor:.4f}"
        for cuant, valor in medidas["deciles"].items()
    )

    lineas.extend(["", "Percentiles (cada 10%):"])
    lineas.extend(
        f"  P{int(cuant * 100)} ({cuant:.2f}): {valor:.4f}"
        for cuant, valor in medidas["percentiles"].items()
    )
    lineas.append(f"  P90 (0.90): {medidas['percentil_90']:.4f}")

    lineas.extend(
        [
            "",
            "Medidas de dispersión:",
            (
                f"  Rango: {medidas['rango']:.4f}\n"
                f"  Varianza: {medidas['varianza']:.4f}\n"
                f"  Desviación típica: {medidas['desv_tipica']:.4f}\n"
                f"  Desviación media: {medidas['desv_media']:.4f}\n"
                f"  Coeficiente de variación: {medidas['coef_var']:.2f}%"
            ),
        ]
    )

    return "\n".join(lineas)


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
    """Punto de entrada de la aplicación gráfica."""

    root = tk.Tk()
    app = EstadisticaGUI(root)
    root.mainloop()


class EstadisticaGUI:
    """Interfaz gráfica principal de la aplicación."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Aplicación de Estadística Descriptiva")
        self.series: Optional[pd.Series] = None

        self._status_var = tk.StringVar(value="No hay datos cargados.")

        self._crear_interfaz()

    def _crear_interfaz(self) -> None:
        instrucciones = (
            "Ingrese datos numéricos separados por espacios o cargue un archivo CSV "
            "para seleccionar una columna numérica."
        )
        ttk.Label(self.root, text=instrucciones, wraplength=500, justify="center").pack(
            padx=10, pady=10
        )

        frame_manual = ttk.LabelFrame(self.root, text="Datos manuales")
        frame_manual.pack(fill="x", padx=10, pady=5)

        self._texto_datos = tk.Text(frame_manual, height=4)
        self._texto_datos.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            frame_manual,
            text="Utilizar datos ingresados",
            command=self._procesar_datos_manual,
        ).pack(padx=10, pady=(0, 10))

        frame_csv = ttk.LabelFrame(self.root, text="Datos desde CSV")
        frame_csv.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            frame_csv,
            text="Seleccionar archivo CSV",
            command=self._cargar_desde_csv,
        ).pack(padx=10, pady=5)

        ttk.Label(self.root, textvariable=self._status_var).pack(padx=10, pady=5)

        frame_acciones = ttk.Frame(self.root)
        frame_acciones.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            frame_acciones,
            text="Calcular estadísticas",
            command=self._mostrar_resultados,
        ).pack(side="left", padx=5)

        ttk.Button(
            frame_acciones,
            text="Generar gráficos",
            command=self._generar_graficos,
        ).pack(side="left", padx=5)

        frame_resultados = ttk.LabelFrame(self.root, text="Resultados")
        frame_resultados.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self._texto_resultados = ScrolledText(frame_resultados, height=15)
        self._texto_resultados.pack(fill="both", expand=True, padx=10, pady=5)
        self._texto_resultados.configure(state="disabled")

    def _procesar_datos_manual(self) -> None:
        try:
            texto = self._texto_datos.get("1.0", "end").strip()
            self.series = parsear_datos_manual(texto)
        except ValueError as exc:
            messagebox.showerror("Datos inválidos", str(exc))
            return

        self._status_var.set(f"Datos manuales cargados ({len(self.series)} valores).")
        messagebox.showinfo("Datos cargados", "Los datos manuales se han cargado correctamente.")

    def _cargar_desde_csv(self) -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
        )
        if not ruta:
            return

        try:
            df = _leer_csv(Path(ruta))
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Error al leer CSV", str(exc))
            return

        columnas_numericas = list(df.select_dtypes(include=["number"]).columns)
        if not columnas_numericas:
            messagebox.showwarning(
                "Sin columnas numéricas",
                "El archivo seleccionado no contiene columnas numéricas.",
            )
            return

        if len(columnas_numericas) == 1:
            columna = columnas_numericas[0]
        else:
            columna = self._mostrar_selector_columnas(columnas_numericas)
            if columna is None:
                return

        try:
            self.series = _clean_numeric_values(df[columna].to_numpy())
        except ValueError as exc:
            messagebox.showerror("Datos insuficientes", str(exc))
            return

        self._status_var.set(
            f"Datos desde '{Path(ruta).name}' - columna '{columna}' ({len(self.series)} valores)."
        )
        messagebox.showinfo("Datos cargados", "Se han cargado los datos desde el CSV seleccionado.")

    def _mostrar_selector_columnas(self, columnas: list[str]) -> Optional[str]:
        dialogo = tk.Toplevel(self.root)
        dialogo.title("Seleccionar columna")
        dialogo.transient(self.root)
        dialogo.grab_set()

        ttk.Label(dialogo, text="Seleccione la columna numérica a utilizar:").pack(
            padx=10, pady=10
        )

        seleccion = tk.StringVar(value=columnas[0])
        combo = ttk.Combobox(dialogo, values=columnas, textvariable=seleccion, state="readonly")
        combo.pack(padx=10, pady=5)
        combo.focus_set()

        resultado: dict[str, Optional[str]] = {"columna": None}

        def confirmar() -> None:
            resultado["columna"] = seleccion.get()
            dialogo.destroy()

        ttk.Button(dialogo, text="Aceptar", command=confirmar).pack(pady=(5, 10))

        self.root.wait_window(dialogo)
        return resultado["columna"]

    def _mostrar_resultados(self) -> None:
        if self.series is None:
            messagebox.showwarning("Sin datos", "Debe cargar datos antes de calcular las estadísticas.")
            return

        medidas = calcular_medidas(self.series)
        texto = formatear_medidas(medidas)
        self._texto_resultados.configure(state="normal")
        self._texto_resultados.delete("1.0", "end")
        self._texto_resultados.insert("1.0", texto)
        self._texto_resultados.configure(state="disabled")

    def _generar_graficos(self) -> None:
        if self.series is None:
            messagebox.showwarning("Sin datos", "Debe cargar datos antes de generar los gráficos.")
            return
        generar_graficos(self.series)


if __name__ == "__main__":
    main()
