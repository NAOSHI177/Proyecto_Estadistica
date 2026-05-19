"""Aplicación interactiva de estadística descriptiva y regresión lineal simple.

Este módulo permite cargar una serie de datos numéricos ya sea de forma manual o
mediante un archivo CSV y calcula múltiples medidas estadísticas descriptivas.
Además genera gráficos descriptivos y gráficos de regresión lineal simple.

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


MIN_DATA_POINTS = 20
EJEMPLO_DESCRIPTIVO = "12 15 18 20 22 24 25 27 29 30 32 35 36 38 40 42 45 47 50 55"
EJEMPLO_REGRESION = "3 5 7 8 11 13 15 16 19 21 22 25 27 28 31 33 35 36 39 41"


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


def parsear_datos_manual(texto: str) -> pd.Series:
    """Convierte una cadena con datos numéricos separados por espacios en serie."""

    if not texto.strip():
        raise ValueError("Debe ingresar al menos un valor numérico.")

    try:
        valores = [float(x.replace(",", ".")) for x in texto.split()]
    except ValueError as exc:  # pragma: no cover - errores de conversión en GUI
        raise ValueError("Los datos ingresados contienen valores no numéricos.") from exc
    return _clean_numeric_values(valores)


def parsear_datos_regresion(texto: str) -> pd.Series:
    """Convierte los datos Y de regresión en una serie numérica."""

    if not texto.strip():
        raise ValueError("Debe ingresar datos extra de regresión.")

    try:
        valores = [float(x.replace(",", ".")) for x in texto.split()]
    except ValueError as exc:  # pragma: no cover - errores de conversión en GUI
        raise ValueError("Los datos extra de regresión contienen valores no numéricos.") from exc

    return pd.Series(valores, dtype="float64").dropna()


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un archivo CSV y valida su existencia."""
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo '{ruta}' no existe.")
    try:
        return pd.read_csv(ruta)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc


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


def calcular_regresion_lineal(x: pd.Series, y: pd.Series) -> dict:
    """Calcula una regresión lineal simple usando NumPy."""

    x_valores = x.astype("float64").to_numpy()
    y_valores = y.astype("float64").to_numpy()

    if np.allclose(x_valores, x_valores[0]):
        raise ValueError("No es posible calcular la regresión porque todos los valores de X son iguales.")

    pendiente, intercepto = np.polyfit(x_valores, y_valores, 1)
    y_pred = pendiente * x_valores + intercepto
    residuos = y_valores - y_pred

    if np.isclose(np.std(x_valores), 0) or np.isclose(np.std(y_valores), 0):
        r = np.nan
    else:
        r = float(np.corrcoef(x_valores, y_valores)[0, 1])

    ss_res = float(np.sum(residuos**2))
    ss_tot = float(np.sum((y_valores - np.mean(y_valores)) ** 2))
    if np.isclose(ss_tot, 0):
        r2 = 1.0 if np.isclose(ss_res, 0) else np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return {
        "pendiente": float(pendiente),
        "intercepto": float(intercepto),
        "ecuacion": f"ŷ = {pendiente:.4f}X + {intercepto:.4f}",
        "r": r,
        "r2": float(r2),
        "y_pred": pd.Series(y_pred),
        "residuos": pd.Series(residuos),
        "cantidad": len(x_valores),
    }


def _formatear_numero(valor: float) -> str:
    """Formatea números y valores no definidos para resultados."""

    return "No definido" if pd.isna(valor) else f"{valor:.4f}"


def formatear_regresion(resultado: dict) -> str:
    """Devuelve una representación textual de la regresión lineal."""

    pendiente = resultado["pendiente"]
    tendencia = "creciente" if pendiente > 0 else "decreciente" if pendiente < 0 else "constante"

    return "\n".join(
        [
            "=== Resultados de regresión lineal ===",
            "",
            f"Cantidad de pares analizados: {resultado['cantidad']}",
            "",
            "Ecuación de la recta:",
            resultado["ecuacion"],
            "",
            f"Pendiente (m): {_formatear_numero(resultado['pendiente'])}",
            f"Intercepto (b): {_formatear_numero(resultado['intercepto'])}",
            f"Coeficiente de correlación de Pearson (r): {_formatear_numero(resultado['r'])}",
            f"Coeficiente de determinación (R²): {_formatear_numero(resultado['r2'])}",
            "",
            "Interpretación:",
            f"- La relación lineal estimada es {tendencia} porque la pendiente es {pendiente:.4f}.",
            "- El valor de R² indica qué proporción de la variabilidad de Y es explicada por X mediante el modelo lineal.",
        ]
    )


def generar_graficos(series: pd.Series) -> None:
    """Genera un histograma y un diagrama de dispersión."""

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


def generar_graficos_regresion(
    x: pd.Series, y: pd.Series, y_pred: pd.Series, residuos: pd.Series
) -> None:
    """Genera gráficos de dispersión con recta de regresión y residuos."""

    x_valores = x.astype("float64").to_numpy()
    y_valores = y.astype("float64").to_numpy()
    y_pred_valores = y_pred.astype("float64").to_numpy()
    residuos_valores = residuos.astype("float64").to_numpy()
    orden = np.argsort(x_valores)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=x_valores, y=y_valores, color="#009688", label="Datos reales")
    plt.plot(x_valores[orden], y_pred_valores[orden], color="#d32f2f", label="Recta estimada")
    plt.title("Dispersión y recta de regresión")
    plt.xlabel("Variable X")
    plt.ylabel("Variable Y")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x_valores, y=residuos_valores, color="#3f51b5")
    plt.axhline(0, color="#d32f2f", linestyle="--", linewidth=1)
    plt.title("Residuos de la regresión")
    plt.xlabel("Variable X")
    plt.ylabel("Residuos")

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
        self.root.title("Aplicación de Estadística Descriptiva y Regresión")
        self.series: Optional[pd.Series] = None
        self.regresion_y: Optional[pd.Series] = None
        self.resultado_regresion: Optional[dict] = None

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

        frame_botones_manual = ttk.Frame(frame_manual)
        frame_botones_manual.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            frame_botones_manual,
            text="Utilizar datos ingresados",
            command=self._procesar_datos_manual,
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            frame_botones_manual,
            text="Cargar datos de ejemplo",
            command=self._cargar_ejemplo_descriptivo,
        ).pack(side="left", padx=5)

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

        frame_regresion = ttk.LabelFrame(self.root, text="Regresión lineal simple")
        frame_regresion.pack(fill="x", padx=10, pady=5)

        texto_regresion = (
            "Se usarán los datos ya cargados como variable X. Ingrese a continuación "
            "los datos correspondientes de la variable Y para calcular la regresión lineal."
        )
        ttk.Label(frame_regresion, text=texto_regresion, wraplength=500, justify="left").pack(
            fill="x", padx=10, pady=(8, 4)
        )

        ttk.Label(frame_regresion, text="Datos Y adicionales para regresión").pack(
            anchor="w", padx=10, pady=(4, 0)
        )

        self._texto_regresion = tk.Text(frame_regresion, height=4)
        self._texto_regresion.pack(fill="x", padx=10, pady=5)

        frame_botones_regresion = ttk.Frame(frame_regresion)
        frame_botones_regresion.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            frame_botones_regresion,
            text="Cargar datos extra de ejemplo",
            command=self._cargar_ejemplo_regresion,
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            frame_botones_regresion,
            text="Calcular regresión lineal",
            command=self._mostrar_resultados_regresion,
        ).pack(side="left", padx=5)

        ttk.Button(
            frame_botones_regresion,
            text="Generar gráficos de regresión",
            command=self._generar_graficos_regresion,
        ).pack(side="left", padx=5)

        frame_resultados = ttk.LabelFrame(self.root, text="Resultados")
        frame_resultados.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self._texto_resultados = ScrolledText(frame_resultados, height=15)
        self._texto_resultados.pack(fill="both", expand=True, padx=10, pady=5)
        self._texto_resultados.configure(state="disabled")

    def _escribir_resultados(self, texto: str) -> None:
        self._texto_resultados.configure(state="normal")
        self._texto_resultados.delete("1.0", "end")
        self._texto_resultados.insert("1.0", texto)
        self._texto_resultados.configure(state="disabled")

    def _cargar_ejemplo_descriptivo(self) -> None:
        self._texto_datos.delete("1.0", "end")
        self._texto_datos.insert("1.0", EJEMPLO_DESCRIPTIVO)

    def _cargar_ejemplo_regresion(self) -> None:
        self._texto_regresion.delete("1.0", "end")
        self._texto_regresion.insert("1.0", EJEMPLO_REGRESION)
        self.resultado_regresion = None

    def _procesar_datos_manual(self) -> None:
        try:
            texto = self._texto_datos.get("1.0", "end").strip()
            self.series = parsear_datos_manual(texto)
        except ValueError as exc:
            messagebox.showerror("Datos inválidos", str(exc))
            return

        self.regresion_y = None
        self.resultado_regresion = None
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

        self.regresion_y = None
        self.resultado_regresion = None
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
        self._escribir_resultados(texto)

    def _generar_graficos(self) -> None:
        if self.series is None:
            messagebox.showwarning("Sin datos", "Debe cargar datos antes de generar los gráficos.")
            return
        self._status_var.set(
            "Mostrando gráficos. Cierre la ventana de gráficos para continuar con la aplicación."
        )
        generar_graficos(self.series)

    def _procesar_datos_extra_regresion(self) -> pd.Series:
        if self.series is None:
            raise ValueError("Debe cargar primero los datos principales antes de calcular la regresión.")

        texto = self._texto_regresion.get("1.0", "end").strip()
        regresion_y = parsear_datos_regresion(texto)

        if len(regresion_y) != len(self.series):
            raise ValueError(
                "La cantidad de datos extra debe coincidir con la cantidad de datos originales cargados."
            )

        if np.allclose(self.series.to_numpy(dtype="float64"), self.series.iloc[0]):
            raise ValueError("No es posible calcular la regresión porque todos los valores de X son iguales.")

        return regresion_y

    def _calcular_regresion_desde_interfaz(self) -> bool:
        try:
            self.regresion_y = self._procesar_datos_extra_regresion()
            self.resultado_regresion = calcular_regresion_lineal(self.series, self.regresion_y)
        except ValueError as exc:
            titulo = "Sin datos principales" if self.series is None else "Regresión inválida"
            messagebox.showwarning(titulo, str(exc))
            return False
        return True

    def _mostrar_resultados_regresion(self) -> None:
        if not self._calcular_regresion_desde_interfaz():
            return

        texto = formatear_regresion(self.resultado_regresion)
        self._escribir_resultados(texto)

    def _generar_graficos_regresion(self) -> None:
        if not self._calcular_regresion_desde_interfaz():
            return

        self._status_var.set(
            "Mostrando gráficos de regresión. Cierre la ventana de gráficos para continuar."
        )
        generar_graficos_regresion(
            self.series,
            self.regresion_y,
            self.resultado_regresion["y_pred"],
            self.resultado_regresion["residuos"],
        )


if __name__ == "__main__":
    main()
