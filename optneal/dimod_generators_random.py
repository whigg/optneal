import numpy as np
import numpy.random

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument


@graph_argument('graph')
def normal(graph, vartype, loc=0.0, scale=1.0, cls=BinaryQuadraticModel,
           seed=None, zero_lbias=False):
    """Generate a bqm with random biases and offset.

    Biases and offset are drawn from a normal distribution range (mean, std).

    Args:
        graph (int/tuple[nodes, edges]/list[edge]/:obj:`~networkx.Graph`):
            The graph to build the bqm on. Either an integer n,
            interpreted as a complete graph of size n, a nodes/edges pair,
            a list of edges or a NetworkX graph.
        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:
            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
        loc (float, optional, default=0.0):
            Mean (“centre”) of the distribution for the random biases.
        scale (float, optional, default=1.0):
            Standard deviation (spread or “width”) of the distribution
            for the random biases. Must be non-negative.
        cls (:class:`.BinaryQuadraticModel`):
            Binary quadratic model class to build from.
        seed (int, optional, default=None):
            Random seed.
        zero_lbias (bool, optional, default=False):
            If true, linear biases will set zero.

    Returns:
        :obj:`.BinaryQuadraticModel`
    """
    if seed is None:
        seed = numpy.random.randint(2**32, dtype=np.uint32)
    r = numpy.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    ldata = np.zeros(len(variables)) if zero_lbias else r.normal(loc, scale, size=len(variables))
    qdata = r.normal(loc, scale, size=len(irow))
    offset = r.normal(loc, scale)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=variables)
