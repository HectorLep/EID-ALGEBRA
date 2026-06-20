# Implementación Simplificada de PageRank 🕸️

Proyecto de Investigación 5 para el curso de **Álgebra Lineal para la Computación** de la **Universidad Católica de Temuco**.

Este repositorio contiene una implementación simplificada del algoritmo **PageRank** mediante el método iterativo, junto con un entorno de experimentación y una interfaz gráfica interactiva para visualizar resultados.

---

## 📂 Estructura del Proyecto

```text
```text
EID-ALGEBRA/
│
├── app.py                     # Interfaz gráfica (Frontend en Streamlit)
├── README.md                  # Documentación principal
├── requirements.txt           # Dependencias de Python
│
├── data/                      # Datos de entrada y salida
│   ├── raw/                   # Archivos JSON y diagramas del ejemplo base
│   └── processed/             # Gráficos e imágenes exportadas
│
├── docs/                      # Documentación académica
│   ├── informe/               # Código fuente en LaTeX del informe escrito
│   └── presentacion/          # Archivos de la presentación final
│
├── notebooks/                 # Entorno de experimentación
│   └── analisis_experimental.ipynb
│
├── src/                       # Código fuente (Backend)
│   ├── pagerank.py            # Lógica matemática y algoritmo core
│   └── visualization.py       # Funciones de renderizado de grafos
│
└── tests/                     # Pruebas automatizadas
    └── test_pagerank.py
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/HectorLep/EID-ALGEBRA.git
cd EID-ALGEBRA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador predeterminado.

---

## 🧪 Pruebas Unitarias

El módulo matemático de PageRank incluye pruebas unitarias para verificar la correcta implementación del algoritmo.

Ejecutar:

```bash
pytest tests/test_pagerank.py -v
```

---

## 📊 Análisis Experimental

El notebook `analisis_experimental.ipynb` contiene experimentos para estudiar la convergencia del algoritmo y analizar el comportamiento del método iterativo.

Para guardar automáticamente el gráfico generado en la carpeta `data/processed/`, agregar antes de `plt.show()`:

```python
plt.savefig(
    "../data/processed/grafico_convergencia.png",
    bbox_inches="tight"
)
```

---

## 📚 Tecnologías Utilizadas

* Python 3
* NumPy
* Pandas
* Matplotlib
* Streamlit
* Pytest

---

## 👥 Autores

Proyecto desarrollado para la asignatura **Álgebra Lineal para la Computación**.

Universidad Católica de Temuco.
