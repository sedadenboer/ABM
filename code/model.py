from mesa import Model, Agent
from mesa.space import NetworkGrid, SingleGrid, ContinuousSpace
from mesa.time import RandomActivation
import networkx as nx
from network import idealised_network
from first_iteration.agents import Individual
from math import sqrt
import matplotlib.pyplot as plt

class Wappie(Agent):
    def __init__(self, unique_id: int, model: Model, pos) -> None:
        super().__init__(unique_id, model)
        self.opinion = self.random.random()
        self.grid_pos = pos
    
    def step(self):
        self.opinion = self.random.random()

class Political_spectrum(Model):
    def __init__(self, width: int, height: int, network_type: str) -> None:
        """A model for people changing opinion on their political beliefs.

        Args:
            width (int): The width of the grid used.
            height (int): The height of the grid used.
            network_type (str): Can be BA, random, idealised or scale_free?
        """
        num_agents = width * height
        self.schedule = RandomActivation(self)

        # create network
        if network_type == "BA":
            self.G = nx.barabasi_albert_graph(n=num_agents, m=2)
        elif network_type == "idealised":
            # self.G = idealised_network(num_agents)
            self.G = nx.random_geometric_graph(num_agents, 0.125)
        elif network_type == "scale_free":
            graph = nx.scale_free_graph(num_agents)
            self.G = nx.Graph(graph)
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
        elif network_type == "random":
            pass
        else:
            ValueError("'network_type' should be either BA, random, idealised or scale_free")

        self.network = NetworkGrid(self.G)

        self.grid = SingleGrid(width, height, torus=True)

        # create agents
        for i in range(num_agents):
            x = i // width
            y = i % width
            grid_pos = (x, y)
            agent = Wappie(i, self, grid_pos)

            self.grid.place_agent(agent, grid_pos)
                
            self.network.place_agent(agent, i)

            self.schedule.add(agent)

        self.running = True

    def step(self):
        self.schedule.step()

if __name__ == "__main__":
    model = Political_spectrum(10, 10, "scale_free")
    nx.draw_kamada_kawai(model.G, node_size=50)
    plt.savefig("../kamada_kawai.png")
