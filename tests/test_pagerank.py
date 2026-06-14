import unittest
import numpy as np
import sys
import os

# Asegurar que src/ esté en el path al ejecutar desde la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pagerank import build_transition_matrix, compute_pagerank, rank_nodes


class TestBuildTransitionMatrix(unittest.TestCase):
    """Pruebas para la construcción de la matriz de transición."""

    def test_columnas_suman_uno(self):
        """Cada columna de la matriz de transición debe sumar exactamente 1."""
        graph = {'A': ['B', 'C'], 'B': ['C'], 'C': ['A']}
        M, nodes = build_transition_matrix(graph)
        col_sums = M.sum(axis=0)
        np.testing.assert_allclose(
            col_sums, np.ones(len(nodes)), atol=1e-10,
            err_msg="Las columnas de la matriz de transición deben sumar 1"
        )

    def test_dangling_node_distribucion_uniforme(self):
        """Un nodo sin enlaces salientes debe distribuir prob. uniforme a todos."""
        graph = {'A': ['B'], 'B': ['A'], 'C': []}  # C es dangling
        M, nodes = build_transition_matrix(graph)
        n = len(nodes)
        c_idx = nodes.index('C')
        expected_col = np.ones(n) / n
        np.testing.assert_allclose(
            M[:, c_idx], expected_col, atol=1e-10,
            err_msg="Columna del dangling node debe ser uniforme (1/n)"
        )

    def test_grafo_simple_2_nodos(self):
        """Matriz correcta para un grafo de 2 nodos con enlace unidireccional."""
        graph = {'A': ['B'], 'B': []}
        M, nodes = build_transition_matrix(graph)
        # B es dangling → columna B es [0.5, 0.5]
        # A apunta a B → columna A es [0, 1] (todo va a B)
        self.assertEqual(M.shape, (2, 2))
        col_sums = M.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(2), atol=1e-10)


class TestComputePagerank(unittest.TestCase):
    """Pruebas para el cálculo completo del algoritmo PageRank."""

    def test_suma_probabilidades_igual_a_uno(self):
        """La suma de todos los puntajes de PageRank debe ser 1.0 (distribución de probabilidad)."""
        graph = {
            'A': ['B', 'C'],
            'B': ['C', 'D'],
            'C': ['A'],
            'D': ['B', 'C']
        }
        result = compute_pagerank(graph, damping_factor=0.85)
        total = sum(result.values())
        self.assertAlmostEqual(
            total, 1.0, places=6,
            msg=f"La suma del PageRank debe ser 1.0, obtenida: {total:.8f}"
        )

    def test_grafo_totalmente_interconectado_4_nodos(self):
        """
        En un grafo donde cada nodo apunta a todos los demás (totalmente
        interconectado), todos los nodos deben tener exactamente 0.25 de puntaje.
        """
        nodes = ['A', 'B', 'C', 'D']
        graph = {n: [m for m in nodes if m != n] for n in nodes}
        result = compute_pagerank(graph, damping_factor=0.85)
        for node, score in result.items():
            self.assertAlmostEqual(
                score, 0.25, places=5,
                msg=f"Nodo {node}: esperado 0.25, obtenido {score:.8f}"
            )

    def test_nodo_aislado_no_falla(self):
        """El algoritmo no debe fallar con un nodo sin ninguna conexión (aislado)."""
        graph = {'A': ['B'], 'B': ['A'], 'C': []}
        try:
            result = compute_pagerank(graph)
        except Exception as e:
            self.fail(f"compute_pagerank lanzó excepción con nodo aislado: {e}")
        self.assertIn('C', result, "El nodo aislado 'C' debe estar en el resultado")

    def test_suma_con_nodo_aislado_igual_a_uno(self):
        """La suma del PageRank también debe ser 1 cuando hay nodos aislados."""
        graph = {'A': ['B'], 'B': ['A'], 'C': []}
        result = compute_pagerank(graph)
        total = sum(result.values())
        self.assertAlmostEqual(
            total, 1.0, places=6,
            msg="La suma del PageRank con nodo aislado debe seguir siendo 1.0"
        )

    def test_resultado_ordenado_de_mayor_a_menor(self):
        """El resultado debe estar ordenado de mayor a menor puntaje."""
        graph = {'A': ['B', 'C', 'D'], 'B': ['A'], 'C': ['A'], 'D': ['A']}
        result = compute_pagerank(graph)
        scores = list(result.values())
        self.assertEqual(
            scores, sorted(scores, reverse=True),
            "Los puntajes deben estar en orden descendente"
        )

    def test_grafo_vacio_retorna_dict_vacio(self):
        """Un grafo vacío debe retornar un diccionario vacío sin errores."""
        result = compute_pagerank({})
        self.assertEqual(result, {}, "Un grafo vacío debe retornar {}")

    def test_nodo_con_muchos_inbound_tiene_mayor_score(self):
        """
        Un nodo al que apuntan todos los demás debe tener el mayor puntaje.
        Este es el principio fundamental de PageRank.
        """
        # A es el 'hub': todos apuntan a A, A apunta a B
        graph = {
            'A': ['B'],
            'B': ['A'],
            'C': ['A'],
            'D': ['A'],
            'E': ['A']
        }
        result = compute_pagerank(graph)
        nodes_by_score = list(result.keys())
        self.assertEqual(
            nodes_by_score[0], 'A',
            f"'A' (nodo hub) debe ser el más importante, obtenido: {nodes_by_score[0]}"
        )

    def test_damping_factor_cero_distribucion_uniforme(self):
        """
        Con damping_factor=0, la ecuación se reduce a r = (1/n)*[1,...,1],
        por lo que todos los nodos deben tener igual puntaje 1/n.
        """
        graph = {'A': ['B'], 'B': ['C'], 'C': ['A'], 'D': ['A']}
        result = compute_pagerank(graph, damping_factor=0.0)
        n = len(result)
        expected = 1.0 / n
        for node, score in result.items():
            self.assertAlmostEqual(
                score, expected, places=5,
                msg=f"Con d=0, nodo {node}: esperado {expected:.6f}, obtenido {score:.8f}"
            )


class TestRankNodes(unittest.TestCase):
    """Pruebas para la función de ordenamiento de nodos."""

    def test_ordenamiento_correcto(self):
        """rank_nodes debe ordenar correctamente de mayor a menor."""
        scores = np.array([0.1, 0.4, 0.3, 0.2])
        nodes = ['A', 'B', 'C', 'D']
        result = rank_nodes(scores, nodes)
        self.assertEqual(list(result.keys()), ['B', 'C', 'D', 'A'])

    def test_valores_correctos_en_resultado(self):
        """Los valores del ranking deben corresponder a los puntajes originales."""
        scores = np.array([0.25, 0.25, 0.25, 0.25])
        nodes = ['A', 'B', 'C', 'D']
        result = rank_nodes(scores, nodes)
        for score in result.values():
            self.assertAlmostEqual(score, 0.25, places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)