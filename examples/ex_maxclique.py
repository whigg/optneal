import optneal as opn

import pyqubo
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 10
    G = nx.random_regular_graph(d=3, n=N)
    print(G.edges)
    # nx.draw_networkx(G)
    # plt.show()

    x = pyqubo.Array.create('x', N, 'BINARY')
    y = pyqubo.Array.create('y', N - 1, 'BINARY')
    ny_sum = np.sum([[(n + 2) * y[n] for n in range(N - 1)]])
    H_cost = 0.5 * ny_sum * (ny_sum - 1) - np.sum([x[u] * x[v] for (u, v) in G.edges])

    model = H_cost.compile()
    bqm = model.to_bqm()
    print(bqm)

    import IPython; IPython.embed()


if __name__ == '__main__':
    main()
