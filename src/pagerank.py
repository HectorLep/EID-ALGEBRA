import numpy as np
from typing import Union


def build_transition_matrix(graph: dict) -> tuple[np.ndarray, list]:
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
) -> tuple[dict, int]:
    
    if not graph:
        return {}, 0

    M, nodes = build_transition_matrix(graph)
    n = len(nodes)

    # Vector inicial: distribución uniforme (suma = 1)
    r = np.ones(n) / n

    # Vector de teletransportación uniforme
    teleport = np.ones(n) / n

    for i in range(max_iterations):
        r_new = damping_factor * M @ r + (1 - damping_factor) * teleport

        # Criterio de convergencia: norma L1 de la diferencia
        if np.linalg.norm(r_new - r, ord=1) < tolerance:
            r = r_new
            break

        r = r_new

    # Retorna el ranking y la iteración exacta en la que se detuvo
    return rank_nodes(r, nodes), i + 1

def rank_nodes(scores: np.ndarray, nodes: list) -> dict:
    node_scores = {node: float(scores[i]) for i, node in enumerate(nodes)}
    return dict(sorted(node_scores.items(), key=lambda x: x[1], reverse=True))