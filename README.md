# Proyecto Estadística Descriptiva

Aplicación interactiva en Python con Tkinter para calcular medidas de estadística
descriptiva, visualizar resultados mediante gráficos y calcular regresión lineal
simple. Permite ingresar los datos principales manualmente o cargarlos desde un
archivo CSV.

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

## Funcionalidades

- Estadística descriptiva.
- Regresión lineal simple usando los datos principales como variable X.
- Carga manual de datos principales.
- Carga de datos principales desde una columna numérica de CSV.
- Botones de ayuda para cargar datos de ejemplo.

## Requisitos de cantidad de datos

- Para estadística descriptiva se requieren al menos 20 datos numéricos válidos.
- Para regresión lineal primero se deben cargar los datos principales.
- Luego se deben ingresar datos extra para la variable Y.
- La cantidad de datos Y extra debe coincidir exactamente con la cantidad de datos
  originales cargados como X.

## Flujo de estadística descriptiva

1. Elige si deseas ingresar los datos manualmente o cargarlos desde un CSV.
2. Si seleccionas CSV, indica la ruta del archivo y la columna numérica a utilizar.
3. También puedes usar el botón **Cargar datos de ejemplo** para rellenar el campo
   manual con 20 valores de prueba.
4. Usa el botón para cargar o utilizar los datos ingresados.
5. Calcula las estadísticas descriptivas:
   - Tendencia central: media, mediana, moda.
   - Posición: cuartiles, deciles, percentiles (incluido el percentil 90).
   - Dispersión: rango, varianza, desviación típica, desviación media y coeficiente de variación.
6. Genera los gráficos descriptivos cuando los necesites.

## Flujo de regresión lineal simple

1. Carga primero los datos principales. Estos datos serán la variable X.
2. En la sección **Regresión lineal simple**, ingresa los datos adicionales de la
   variable Y.
3. Puedes usar el botón **Cargar datos extra de ejemplo** para rellenar solo el
   campo Y con 20 valores de prueba.
4. Presiona **Calcular regresión lineal** para obtener:
   - Ecuación de la recta.
   - Pendiente.
   - Intercepto.
   - Coeficiente de correlación de Pearson.
   - Coeficiente de determinación.
5. Presiona **Generar gráficos de regresión** para ver la dispersión con recta de
   regresión y el gráfico de residuos.

## Gráficos disponibles

- Histograma.
- Diagrama de dispersión descriptivo.
- Dispersión con recta de regresión.
- Gráfico de residuos.

## Notas

- Es necesario proporcionar al menos 20 datos numéricos válidos para los datos principales.
- Al trabajar con CSV se detectan automáticamente las columnas numéricas disponibles.
- Para regresión lineal, si se cargan 25 datos principales, se deben ingresar
  exactamente 25 datos Y.
- Cierra la ventana de gráficos para continuar usando la aplicación.
