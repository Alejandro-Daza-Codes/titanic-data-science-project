# titanic-data-science-project
Clasificación de supervivencia en el Titanic. Proyecto de ML con preprocesamiento (imputación, One-Hot) y comparación de 5 modelos (DT, RF, NB, SVM, MLP). El análisis confirmó a 'Sexo' y 'Clase' como predictores clave. Random Forest es el mejor modelo, logrando un 83.43% de exactitud en la predicción.
# 🚢 Titanic Survival Prediction: Comparación de Modelos de Clasificación

## 📝 Descripción del Proyecto

Este repositorio contiene un proyecto de **Machine Learning** enfocado en predecir la supervivencia de los pasajeros del RMS Titanic utilizando el dataset clásico de Kaggle.

El objetivo principal fue **comparar el rendimiento** de cinco algoritmos de clasificación, aplicando un riguroso proceso de **Análisis Exploratorio de Datos (EDA)** y **Preprocesamiento** para optimizar la capacidad de generalización de los modelos.

## 📊 Resultados y Conclusión (Benchmark)

Se entrenaron y evaluaron cinco modelos en diferentes proporciones de *split* (90/10 y 60/40) para identificar el más robusto.

El modelo **Random Forest (RF)** demostró el mejor rendimiento en el *split* 60/40, confirmándose como el algoritmo más efectivo para este problema de clasificación binaria.

| Algoritmo de Clasificación | Exactitud (100%) | Exactitud (90/10) | Exactitud (60/40) |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | 0.8830 | 0.7865 | **0.8343** |
| Support Vector Machine (SVM) | 0.8459 | 0.8090 | **0.8343** |
| Naive Bayes (NB) | 0.7885 | 0.8202 | 0.8315 |
| Árbol de Decisión (DT) | 0.8785 | 0.7865 | 0.8202 |
| Red Neuronal (MLP) | 0.8628 | 0.7865 | 0.8006 |

### Conclusiones Clave del Modelo Ganador (Random Forest)

El análisis de la **Matriz de Confusión** del modelo RF (60/40) mostró:
* **Falsos Negativos (FN) Optimizados:** Se minimizó el error más costoso (predecir la muerte de un pasajero que realmente sobrevivió), un factor crucial para métricas como el *Recall* y el *F1-Score*.
* **Predictores Dominantes:** El EDA y el análisis de la importancia de características (Feature Importance) del RF confirmaron que el **Género** (`Sex_male`) y la **Clase del Billete** (`Pclass` / `Fare`) son los factores más influyentes en la supervivencia.

## ⚙️ Metodología y Ejecución

### Archivos
* **`CASO TITANIC.ipynb`**: Cuaderno principal de Colab con todo el proceso (EDA, Preprocesamiento, Entrenamiento y Visualizaciones).
* **`train.csv`**: El dataset original utilizado para el entrenamiento y la evaluación.

### Pasos Clave del Preprocesamiento

1.  **Gestión de Valores Faltantes (NaNs):**
    * `Age`: Imputada con la media (29.70).
    * `Cabin`: Eliminada debido a la alta tasa de datos faltantes (más del 77%).
    * `Embarked`: Filas eliminadas (solo 2 NaNs).
2.  **Eliminación de Variables:** Se descartaron `Name`, `PassengerId` y `Ticket` por su nulo valor predictivo.
3.  **Codificación:** Las variables categóricas (`Sex` y `Embarked`) se transformaron usando **One-Hot Encoding** (`pd.get_dummies`).
4.  **Estandarización:** Se aplicó **StandardScaler** a las variables numéricas para normalizar los datos, paso esencial para SVM y Redes Neuronales.

### Visualizaciones de Modelos
Se incluyeron visualizaciones avanzadas para interpretar los modelos:
* **Árbol de Decisión:** Gráfico del árbol para ver las reglas explícitas (confirmando a `Sex_male` y `Pclass` como nodos principales).
* **Random Forest:** Gráfico de importancia de características (*Feature Importance*).
* **Naive Bayes:** Gráfico de distribución Gaussiana, mostrando la superposición de densidades de probabilidad.
* **Linear SVM (Proxy):** Gráfico de coeficientes, que actúa como un mapa de pesos para las características.

---
