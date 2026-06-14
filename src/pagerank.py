import numpy as np
from typing import Union


def build_transition_matrix(graph: dict) -> tuple[np.ndarray, list]:
    """
    Construye la matriz de transición probabilística a partir de un grafo.

    La matriz de transición M tiene la propiedad de que cada columna suma 1
    (matriz estocástica por columnas). El elemento M[i][j] representa la
    probabilidad de ir desde el nodo j hacia el nodo i.

    Los "dangling nodes" (nodos sin enlaces salientes) distribuyen su
    probabilidad de forma uniforme entre todos los nodos.

    Args:
        graph (dict): Diccionario donde las claves son nodos y los valores son
                      listas de nodos a los que apuntan (enlaces salientes).
                      Ejemplo: {'A': ['B', 'C'], 'B': ['A'], 'C': []}

    Returns:
        tuple: (matriz_transicion, lista_nodos)
            - matriz_transicion: np.ndarray de forma (n, n)
            - lista_nodos: lista con los nodos en el orden de la matriz
    """
    nodes = sorted(graph.keys())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    M = np.zeros((n, n))

    for node in nodes:
        col = node_index[node]
        out_links = graph[node]

        # Dangling node: sin enlaces salientes → distribución uniforme
        if len(out_links) == 0:
            M[:, col] = 1.0 / n
        else:
            prob = 1.0 / len(out_links)
            for target in out_links:
                if target in node_index:
                    row = node_index[target]
                    M[row, col] = prob

    return M, nodes


def compute_pagerank(
    graph: dict,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> dict:
    """
    Calcula el PageRank de cada nodo usando el método iterativo de potencias.

    La fórmula iterativa es:
        r(t+1) = d * M * r(t) + (1 - d) / n * [1, 1, ..., 1]^T

    Donde:
        r(t) es el vector de ranking en la iteración t
        M    es la matriz de transición
        d    es el damping factor (factor de amortiguamiento)
        n    es el número de nodos

    La iteración se detiene cuando se alcanza convergencia (la diferencia entre
    dos iteraciones consecutivas es menor que la tolerancia) o se llega al
    número máximo de iteraciones.

    Args:
        graph (dict): Diccionario del grafo (ver build_transition_matrix).
        damping_factor (float): Factor de amortiguamiento d, usualmente 0.85.
                                 Representa la probabilidad de seguir un enlace
                                 (vs. saltar a una página aleatoria).
        max_iterations (int): Número máximo de iteraciones permitidas.
        tolerance (float): Umbral de convergencia (norma L1 de la diferencia).

    Returns:
        dict: Diccionario {nodo: puntaje_pagerank} ordenado de mayor a menor.
    """
    if not graph:
        return {}

    M, nodes = build_transition_matrix(graph)
    n = len(nodes)

    # Vector inicial: distribución uniforme (suma = 1)
    r = np.ones(n) / n

    # Vector de teletransportación uniforme
    teleport = np.ones(n) / n

    for _ in range(max_iterations):
        r_new = damping_factor * M @ r + (1 - damping_factor) * teleport

        # Criterio de convergencia: norma L1 de la diferencia
        if np.linalg.norm(r_new - r, ord=1) < tolerance:
            r = r_new
            break

        r = r_new

    return rank_nodes(r, nodes)


def rank_nodes(scores: np.ndarray, nodes: list) -> dict:
    """
    Asocia los puntajes finales a sus nodos correspondientes y los ordena
    de mayor a menor importancia.

    Args:
        scores (np.ndarray): Vector con los puntajes de PageRank.
        nodes (list): Lista de nodos en el mismo orden que el vector.

    Returns:
        dict: Diccionario ordenado {nodo: puntaje} de mayor a menor.
    """
    node_scores = {node: float(scores[i]) for i, node in enumerate(nodes)}
    return dict(sorted(node_scores.items(), key=lambda x: x[1], reverse=True))