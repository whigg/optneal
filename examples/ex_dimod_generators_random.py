import networkx as nx
import dimod

from optneal.dimod_generators_random import normal


def main():
    K_7 = nx.complete_graph(7)
    bqm = normal(K_7, dimod.SPIN, loc=0.0, scale=1.0, zero_lbias=True)
    print(bqm)


if __name__ == '__main__':
    main()
