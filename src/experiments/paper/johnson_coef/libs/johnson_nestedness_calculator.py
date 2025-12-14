# johnson_nestedness_calculator.py

import numpy as np


class JohnsonNestednessCalculator(object):
    """
    Calcula a nestedness segundo Johnson et al. (2013) para
    matrizes de adjacência (unimodais) ou biadjacentes (bipartidas) binárias.

    Se a matriz for retangular (n_rows != n_cols), ela é interpretada como
    matriz biadjacente de um grafo bipartido (camada 0 = linhas, camada 1 = colunas).

    A definição segue (adaptada para bipartido):

        - Constrói-se a matriz de adjacência completa A:
              A = [[0,   M   ],
                   [M^T, 0   ]]

        - Grau de cada nó: k_i = sum_j A_{ij}
        - Número de vizinhos partilhados: (A^2)_{ij}
        - Nestedness por par:
              g_ij = (A^2)_{ij} / (k_i * k_j), i != j

        - Nestedness global bruto:
              g_tilde = (1 / (N*(N-1))) * sum_{i != j} g_ij

        - Valor esperado no modelo de configuração:
              g_conf = <k^2> / (<k>^2 * N)

        - Nestedness global normalizado:
              g = g_tilde / g_conf

        - Nestedness local do nó i:
              g_i_tilde = (1/(N-1)) * sum_{j != i} g_ij
              g_i = g_i_tilde / g_conf
    """

    def __init__(self, mat):
        """
        :param mat: matriz binária (numpy.array) 2D.
                    Pode ser quadrada (adjacência) ou retangular (biadjacente).
        """
        mat = np.asarray(mat)
        assert mat.ndim == 2, "Matriz deve ser 2D."
        self.check_input_matrix_is_binary(mat)
        self.check_degrees_nonzero(mat)

        self.original_mat = mat
        self.is_bipartite = (mat.shape[0] != mat.shape[1])

        # Constrói matriz de adjacência completa A
        self.A = self._build_full_adjacency(mat)
        self.N = self.A.shape[0]

        # Graus
        self.k = self.A.sum(axis=1).astype(float)

    @staticmethod
    def check_input_matrix_is_binary(mat):
        """Verifica se a matriz é binária (0/1)."""
        if not np.all(np.logical_or(mat == 0, mat == 1)):
            raise AssertionError("Input matrix is not binary.")

    @staticmethod
    def check_degrees_nonzero(mat):
        """
        Garante que não existam linhas/colunas totalmente zero
        (isso deve ser verdade se você já filtrou antes).
        """
        if np.any(mat.sum(axis=1) == 0):
            raise AssertionError("Matrix has rows with only zeros.")
        if np.any(mat.sum(axis=0) == 0):
            raise AssertionError("Matrix has columns with only zeros.")

    def _build_full_adjacency(self, mat):
        """
        Constrói matriz de adjacência completa a partir de:
        - matriz quadrada: assume adjacência já (simetriza).
        - matriz retangular: assume biadjacente (grafo bipartido).
        """
        mat = np.asarray(mat, dtype=int)
        n_rows, n_cols = mat.shape

        if n_rows == n_cols:
            # Rede unimodal: garante simetria (assume grafo não-direcionado)
            A = np.where(mat + mat.T > 0, 1, 0).astype(int)
        else:
            # Rede bipartida: constrói bloco [0, M; M^T, 0]
            zero_rows = np.zeros((n_rows, n_rows), dtype=int)
            zero_cols = np.zeros((n_cols, n_cols), dtype=int)
            A = np.block([[zero_rows, mat],
                          [mat.T,    zero_cols]])
        return A

    def nestedness(self, return_local=False):
        """
        Calcula nestedness global e, opcionalmente, local.

        :param return_local: se True, retorna também nestedness local por nó.
        :return: dict com:
            - 'g_raw':       nestedness global bruto  (tilde g)
            - 'g_conf':      nestedness esperado no modelo de configuração
            - 'g_norm':      nestedness global normalizado (g_raw / g_conf)
            - 'local_raw':   array (N,) com nestedness local bruto (ou None)
            - 'local_norm':  array (N,) com nestedness local normalizado (ou None)
        """
        A = self.A
        N = self.N
        k = self.k

        # (A^2)_{ij} = número de vizinhos partilhados
        B = A.dot(A).astype(float)

        # g_ij = B_ij / (k_i * k_j), para i != j
        ki = k.reshape(N, 1)
        kj = k.reshape(1, N)
        denom = ki * kj

        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.zeros_like(B, dtype=float)
            mask = denom > 0
            G[mask] = B[mask] / denom[mask]

        # zera diagonal (i == j não entra)
        np.fill_diagonal(G, 0.0)

        # Nestedness global bruto
        num_pairs = N * (N - 1)
        if num_pairs > 0:
            g_raw = G.sum() / num_pairs
        else:
            g_raw = np.nan

        # Valor esperado no modelo de configuração
        k_mean = k.mean()
        k_sq_mean = np.mean(k ** 2)

        if k_mean == 0 or N == 0:
            g_conf = np.nan
            g_norm = np.nan
        else:
            g_conf = (k_sq_mean / (k_mean ** 2)) / float(N)
            g_norm = g_raw / g_conf if g_conf > 0 else np.nan

        local_raw = None
        local_norm = None

        if return_local:
            if N > 1:
                local_raw = G.sum(axis=1) / (N - 1)
            else:
                local_raw = np.full(N, np.nan, dtype=float) 
            local_norm = local_raw / \
                g_conf if g_conf > 0 else np.full_like(local_raw, np.nan)

        return {
            "g_raw": g_raw,
            "g_conf": g_conf,
            "g_norm": g_norm,
            "local_raw": local_raw,
            "local_norm": local_norm,
        }
