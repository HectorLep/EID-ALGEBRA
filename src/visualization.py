import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

try:
    from .pagerank import build_transition_matrix
except ImportError:
    # Permite ejecutar este archivo como script suelto (ej. desde test_visual_temp.py)
    # sin que falle por no encontrar un paquete padre
    from pagerank import build_transition_matrix


def build_networkx_graph(graph: dict) -> nx.DiGraph:
    """
    Convierte el diccionario {nodo: [destinos]} usado por pagerank.py
    en un grafo dirigido de NetworkX.
    """
    G = nx.DiGraph()
    G.add_nodes_from(graph.keys())

    for origen, destinos in graph.items():
        for destino in destinos:
            G.add_edge(origen, destino)

    return G


def draw_graph(graph: dict, scores: dict = None, ax=None, seed: int = 42):
    """
    Dibuja el grafo dirigido. Si se entregan scores (salida de compute_pagerank),
    el tamaño de cada nodo se escala según su importancia relativa.
    """
    G = build_networkx_graph(graph)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    pos = nx.spring_layout(G, seed=seed, k=0.8)

    if scores:
        max_score = max(scores.values())
        # Tamaño mínimo más alto (900) para que nombres largos como "Fan Blog"
        # entren dentro del círculo incluso con score bajo
        node_sizes = [2600 * (scores.get(n, 0) / max_score) + 900 for n in G.nodes()]
        node_values = [scores.get(n, 0) / max_score for n in G.nodes()]
        cmap = plt.cm.Greens
        # Recortamos el rango del cmap (0.25 a 0.85) para evitar tanto el
        # verde casi blanco (score bajo, se confunde con el fondo) como
        # el verde casi negro (score alto, no haría falta texto blanco)
        node_colors = [cmap(0.25 + 0.6 * v) for v in node_values]
        # Texto negro sobre verde claro (score bajo) y blanco sobre verde oscuro
        # (score alto), para que la etiqueta se lea bien en ambos extremos
        label_colors = {n: ("black" if v < 0.5 else "white") for n, v in zip(G.nodes(), node_values)}
    else:
        node_sizes = [1800 for _ in G.nodes()]
        node_colors = "skyblue"
        label_colors = {n: "black" for n in G.nodes()}

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.3,
    )

    # Dibujamos las etiquetas nodo por nodo para poder variar el color
    # y achicar la fuente automáticamente si el nombre es largo
    for nodo in G.nodes():
        x, y = pos[nodo]
        tamano_fuente = 9 if len(nodo) <= 8 else 7.5
        ax.text(
            x, y, nodo,
            fontsize=tamano_fuente, fontweight="bold",
            ha="center", va="center",
            color=label_colors[nodo],
        )

    # Separamos visualmente cada enlace según si existe el enlace opuesto.
    # Si A->B y B->A coexisten, cada flecha se curva hacia un lado distinto
    # para que no queden superpuestas; si el enlace es de ida única, casi recta.
    for origen, destino in G.edges():
        si_tiene_opuesto = G.has_edge(destino, origen)
        rad = 0.18 if si_tiene_opuesto else 0.05

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(origen, destino)],
            arrowstyle="-|>", arrowsize=18,
            width=1.4,
            connectionstyle=f"arc3,rad={rad}",
            edge_color="gray",
            node_size=node_sizes if scores else 1800,
        )

    ax.set_title("Grafo dirigido de la red")
    ax.axis("off")

    # Expandimos manualmente los límites del eje según las posiciones reales,
    # para que ningún nodo (especialmente los de nombre largo) quede cortado
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    padding_x = max((max(xs) - min(xs)) * 0.25, 0.55)
    padding_y = max((max(ys) - min(ys)) * 0.25, 0.55)
    ax.set_xlim(min(xs) - padding_x, max(xs) + padding_x)
    ax.set_ylim(min(ys) - padding_y, max(ys) + padding_y)

    return ax


def compute_pagerank_history(
    graph: dict,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> tuple[list, list]:
    """
    Réplica del bucle iterativo de pagerank.compute_pagerank, pero
    guardando el vector r en cada paso. Se usa solo para graficar la
    convergencia; no reemplaza ni modifica la función original de Héctor,
    que sigue siendo la fuente de verdad para el ranking final.

    Retorna:
        history: lista de vectores r (uno por iteración, incluye r0)
        nodes: lista de nodos en el mismo orden que las columnas de history
    """
    if not graph:
        return [], []

    M, nodes = build_transition_matrix(graph)
    n = len(nodes)

    r = np.ones(n) / n
    teleport = np.ones(n) / n

    history = [r.copy()]

    for _ in range(max_iterations):
        r_new = damping_factor * M @ r + (1 - damping_factor) * teleport
        history.append(r_new.copy())

        if np.linalg.norm(r_new - r, ord=1) < tolerance:
            break

        r = r_new

    return history, nodes


def plot_convergence(graph: dict, damping_factor: float = 0.85, ax=None,
                      max_iterations: int = 100, tolerance: float = 1e-8):
    """
    Grafica la evolución del score de cada nodo a lo largo de las
    iteraciones de PageRank, hasta llegar a la convergencia.
    """
    history, nodes = compute_pagerank_history(
        graph, damping_factor=damping_factor,
        max_iterations=max_iterations, tolerance=tolerance
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    history_array = np.array(history)  # shape: (n_iter, n_nodos)

    # Ciclamos estilos de línea y marcador para que dos curvas con valores
    # casi idénticos (ej. nodos simétricos en la red) se sigan distinguiendo
    estilos_linea = ["-", "--", "-.", ":"]
    marcadores = ["o", "s", "^", "D", "v"]

    for i, nodo in enumerate(nodes):
        ax.plot(
            history_array[:, i],
            linestyle=estilos_linea[i % len(estilos_linea)],
            marker=marcadores[i % len(marcadores)],
            markersize=4,
            linewidth=1.6,
            label=nodo,
        )

    ax.set_xlabel("Iteración")
    ax.set_ylabel("Score PageRank")
    ax.set_title(f"Convergencia del algoritmo (damping = {damping_factor})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    return ax