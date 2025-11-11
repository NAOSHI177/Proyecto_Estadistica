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

        self._resultado_vars: dict[str, tk.StringVar] = {}
        self._resultado_listas: dict[str, ttk.Treeview] = {}
        self._crear_frames_resultados(frame_resultados)

    def _crear_frames_resultados(self, contenedor: ttk.LabelFrame) -> None:
        frame_medidas = ttk.Frame(contenedor)
        frame_medidas.pack(fill="x", padx=10, pady=5)

        info_escalar = [
            ("media", "Media"),
            ("mediana", "Mediana"),
            ("moda", "Moda"),
            ("percentil_90", "Percentil 90"),
            ("rango", "Rango"),
            ("varianza", "Varianza"),
            ("desv_tipica", "Desviación típica"),
            ("desv_media", "Desviación media"),
            ("coef_var", "Coeficiente de variación"),
        ]

        columnas = 3
        for col in range(columnas):
            frame_medidas.columnconfigure(col, weight=1)

        for idx, (clave, titulo) in enumerate(info_escalar):
            frame = ttk.LabelFrame(frame_medidas, text=titulo)
            frame.grid(row=idx // columnas, column=idx % columnas, padx=5, pady=5, sticky="nsew")

            variable = tk.StringVar(value="Sin calcular")
            ttk.Label(frame, textvariable=variable, anchor="w").pack(fill="x", padx=5, pady=5)
            self._resultado_vars[clave] = variable

        frame_listas = ttk.Frame(contenedor)
        frame_listas.pack(fill="both", expand=True, padx=10, pady=5)

        info_listas = [
            ("cuartiles", "Cuartiles"),
            ("deciles", "Deciles"),
            ("percentiles", "Percentiles (cada 10%)"),
        ]

        for clave, titulo in info_listas:
            frame = ttk.LabelFrame(frame_listas, text=titulo)
            frame.pack(fill="both", expand=True, padx=5, pady=5)

            tree = ttk.Treeview(frame, columns=("cuantil", "valor"), show="headings", height=4)
            tree.heading("cuantil", text="Cuantil")
            tree.heading("valor", text="Valor")
            tree.column("cuantil", anchor="center", width=130)
            tree.column("valor", anchor="center", width=130)

            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            scrollbar.pack(side="right", fill="y")

            self._resultado_listas[clave] = tree

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
        self._actualizar_frames_resultados(medidas)

    def _actualizar_frames_resultados(self, medidas: dict) -> None:
        def _formatear_valor(clave: str, valor: float) -> str:
            if pd.isna(valor):
                return "No disponible"
            if clave == "coef_var":
                return f"{valor:.2f} %"
            return f"{valor:.4f}"

        for clave, variable in self._resultado_vars.items():
            valor = medidas.get(clave, np.nan)
            variable.set(_formatear_valor(clave, valor))

        for clave, tree in self._resultado_listas.items():
            for item in tree.get_children():
                tree.delete(item)

            serie = medidas.get(clave)
            if serie is None:
                continue

            for cuantil, valor in serie.items():
                etiqueta = self._formatear_cuantil(clave, cuantil)
                texto_valor = "No disponible" if pd.isna(valor) else f"{valor:.4f}"
                tree.insert("", "end", values=(etiqueta, texto_valor))

    @staticmethod
    def _formatear_cuantil(clave: str, cuantil: float) -> str:
        if clave == "cuartiles":
            return f"Q{int(cuantil * 4)} ({cuantil:.2f})"
        if clave == "deciles":
            return f"D{int(cuantil * 10)} ({cuantil:.1f})"
        if clave == "percentiles":
            return f"P{int(cuantil * 100)} ({cuantil:.0f}%)"
        return f"{cuantil:.2f}"

    def _generar_graficos(self) -> None:
        if self.series is None:
            messagebox.showwarning("Sin datos", "Debe cargar datos antes de generar los gráficos.")
            return
        self._status_var.set(
            "Mostrando gráficos. Cierre la ventana de gráficos para continuar con la aplicación."
        )
        generar_graficos(self.series)


if __name__ == "__main__":
    main()
