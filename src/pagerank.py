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
    tolerance: float = 1e-8,
    return_history: bool = False 
):
    
    if not graph:
        if return_history:
            return {}, 0, [], []
        return {}, 0

    M, nodes = build_transition_matrix(graph)
    n = len(nodes)

    r = np.ones(n) / n
    teleport = np.ones(n) / n

    # NUEVO: Guardamos el estado inicial si nos piden el historial
    history = [r.copy()] if return_history else None

    for i in range(max_iterations):
        r_new = damping_factor * M @ r + (1 - damping_factor) * teleport

        if return_history:
            history.append(r_new.copy())

        # Criterio de convergencia
        if np.linalg.norm(r_new - r, ord=1) < tolerance:
            r = r_new
            break

        r = r_new

    # NUEVO: Si piden el historial, devolvemos 4 cosas. Si no, solo las 2 de siempre.
    if return_history:
        return rank_nodes(r, nodes), i + 1, history, nodes
        
    return rank_nodes(r, nodes), i + 1

def rank_nodes(scores: np.ndarray, nodes: list) -> dict:
    node_scores = {node: float(scores[i]) for i, node in enumerate(nodes)}
    return dict(sorted(node_scores.items(), key=lambda x: x[1], reverse=True))