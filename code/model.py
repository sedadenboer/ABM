from mesa import Model
import networkx as nx
from agents import Wappie

class Political_spectrum(Model):
    def __init__(self, num_agents) -> None:

        # create grid and network
        nx.barabasi_albert_graph(n=num_agents, m=2)

        # create agents

        # 
        pass

    def step(self):
        pass