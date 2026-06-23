"""
Implementación Simplificada de PageRank — Interfaz Streamlit
Proyecto de Investigación 5 — Álgebra Lineal para la Computación

Esta app integra de forma reactiva el algoritmo de PageRank (src/pagerank.py)
con las visualizaciones del grafo y la convergencia (src/visualization.py),
de acuerdo a lo solicitado en la pauta para la implementación computacional
(sección III.A) del proyecto.
"""
import json
import streamlit as st
import matplotlib.pyplot as plt

from src.pagerank import compute_pagerank, build_transition_matrix
from src.visualization import draw_graph, plot_convergence


# --------------------------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------------------------
def json_a_grafo(data: dict) -> dict:
    """
    Convierte el formato {nodos: [...], enlaces: [{origen, destino}, ...]}
    (usado en data/raw/ejemplo_base.json) al formato {nodo: [destinos]}
    que espera compute_pagerank.
    """
    grafo = {nodo: [] for nodo in data["nodos"]}
    for enlace in data["enlaces"]:
        grafo[enlace["origen"]].append(enlace["destino"])
    return grafo


@st.cache_data
def cargar_ejemplo_base() -> dict:
    with open("data/raw/ejemplo_base.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_a_grafo(data)


# --------------------------------------------------------------------
# Configuración general de la página
# --------------------------------------------------------------------
st.set_page_config(page_title="PageRank — Álgebra Lineal", layout="wide")

st.title("Implementación Simplificada de PageRank")
st.markdown(
    "Proyecto de Investigación 5 — Álgebra Lineal para la Computación. "
    "Esta app calcula el ranking de un grafo dirigido utilizando el "
    "algoritmo PageRank y permite analizar el efecto del factor de "
    "amortiguamiento sobre la convergencia."
)

# --------------------------------------------------------------------
# Carga de datos: ejemplo base del repo, o un JSON propio del usuario
# --------------------------------------------------------------------
st.sidebar.header("Datos de entrada")

fuente = st.sidebar.radio(
    "Selecciona la red a analizar",
    ["Ejemplo base (YouTube y redes asociadas)", "Cargar archivo JSON propio"],
)

if fuente == "Ejemplo base (YouTube y redes asociadas)":
    grafo = cargar_ejemplo_base()
else:
    archivo = st.sidebar.file_uploader(
        "Archivo JSON con la estructura {nodos: [...], enlaces: [{origen, destino}]}",
        type="json",
    )
    if archivo is not None:
        data = json.load(archivo)
        grafo = json_a_grafo(data)
    else:
        st.info("Sube un archivo JSON para comenzar, o vuelve al ejemplo base.")
        st.stop()

# --------------------------------------------------------------------
# Controles reactivos: factor de amortiguamiento y máximo de iteraciones
# --------------------------------------------------------------------
st.sidebar.header("Parámetros del algoritmo")

damping_factor = st.sidebar.slider(
    "Factor de amortiguamiento (damping factor)",
    min_value=0.50, max_value=0.99, value=0.85, step=0.01,
    help="Probabilidad de que el 'navegante aleatorio' siga un enlace "
         "en vez de saltar a una página al azar. El valor estándar usado "
         "por Google es 0.85.",
)

max_iterations = st.sidebar.slider(
    "Máximo de iteraciones", min_value=10, max_value=500, value=100, step=10,
)

# --------------------------------------------------------------------
# Cálculo del ranking (algoritmo de Héctor, sin modificaciones)
# --------------------------------------------------------------------
scores, iteraciones_usadas = compute_pagerank(
    grafo, damping_factor=damping_factor, max_iterations=max_iterations
)

# --------------------------------------------------------------------
# Resultados
# --------------------------------------------------------------------
col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("Grafo dirigido de la red")
    fig_grafo, ax_grafo = plt.subplots(figsize=(7, 6))
    draw_graph(grafo, scores=scores, ax=ax_grafo)
    st.pyplot(fig_grafo)

with col_der:
    st.subheader("Ranking de nodos")
    tabla_ranking = [
        {"Posición": i + 1, "Nodo": nodo, "Score PageRank": round(score, 4)}
        for i, (nodo, score) in enumerate(scores.items())
    ]
    st.table(tabla_ranking)
    st.metric("Iteraciones hasta convergencia", iteraciones_usadas)

st.subheader("Convergencia del algoritmo")
fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
plot_convergence(
    grafo, damping_factor=damping_factor,
    ax=ax_conv, max_iterations=max_iterations,
)
st.pyplot(fig_conv)

# --------------------------------------------------------------------
# Sección extra: matriz de transición y exploración paso a paso
# de las iteraciones, para apoyar la explicación matemática en vivo
# durante la presentación (sin necesidad de escribir en la pizarra).
# --------------------------------------------------------------------
st.divider()
st.subheader("Matriz de transición y proceso iterativo")
st.markdown(
    "Esta sección muestra cómo se construye la matriz de transición a "
    "partir del grafo, y cómo evoluciona el vector de probabilidades "
    "en cada iteración hasta alcanzar la convergencia."
)

M, nodes_M = build_transition_matrix(grafo)

with st.expander("Ver matriz de transición M (columna = origen, fila = destino)", expanded=True):
    tabla_M = []
    for i, nodo_fila in enumerate(nodes_M):
        fila = {"": nodo_fila}
        for j, nodo_col in enumerate(nodes_M):
            fila[nodo_col] = round(float(M[i, j]), 3)
        tabla_M.append(fila)
    st.table(tabla_M)
    st.caption(
        "Cada columna representa los enlaces salientes de un nodo, "
        "repartidos en partes iguales entre sus destinos. Cada columna suma 1."
    )

_, _, history, nodes_hist = compute_pagerank(
    grafo, damping_factor=damping_factor, max_iterations=max_iterations,
    return_history=True,
)

st.markdown("**Explorar el vector de probabilidades en cada iteración:**")
iteracion_seleccionada = st.slider(
    "Selecciona la iteración a visualizar",
    min_value=0, max_value=len(history) - 1, value=0,
)

vector_actual = history[iteracion_seleccionada]
vector_anterior = history[iteracion_seleccionada - 1] if iteracion_seleccionada > 0 else None

tabla_iteracion = []
for i, nodo in enumerate(nodes_hist):
    fila = {"Nodo": nodo, f"Score en iteración {iteracion_seleccionada}": round(float(vector_actual[i]), 4)}
    if vector_anterior is not None:
        fila["Cambio respecto a la iteración anterior"] = round(float(vector_actual[i] - vector_anterior[i]), 4)
    tabla_iteracion.append(fila)

st.table(tabla_iteracion)

if iteracion_seleccionada == 0:
    st.caption("Iteración 0: distribución inicial uniforme (1/n para cada nodo).")
elif iteracion_seleccionada == len(history) - 1:
    st.caption("Esta es la última iteración calculada (estado de convergencia o máximo alcanzado).")