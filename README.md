# Proyecto Estadística Descriptiva

Aplicación interactiva en Python para calcular medidas de estadística descriptiva y
visualizar los resultados mediante gráficos. Permite ingresar los datos manualmente o
cargarlos desde un archivo CSV.

## Requisitos

- Python 3.9 o superior.
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `seaborn`.

Puedes instalarlas con:

```bash
pip install -r requirements.txt
```

o manualmente:

```bash
pip install numpy pandas matplotlib seaborn
```

## Uso

Ejecuta el script principal desde la terminal:

```bash
python stats_app.py
```

1. Elige si deseas ingresar los datos manualmente o cargarlos desde un CSV.
2. Si seleccionas CSV, indica la ruta del archivo y la columna numérica a utilizar.
3. La aplicación mostrará en consola las medidas calculadas:
   - Tendencia central: media, mediana, moda.
   - Posición: cuartiles, deciles, percentiles (incluido el percentil 90).
   - Dispersión: rango, varianza, desviación típica, desviación media y coeficiente de variación.
4. Finalmente se generarán un histograma de frecuencias y un diagrama de dispersión.

## Notas

- Es necesario proporcionar al menos 10 datos numéricos válidos.
- Al trabajar con CSV se detectan automáticamente las columnas numéricas disponibles.
- Cierra la ventana de gráficos para finalizar la ejecución del programa.
