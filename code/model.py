from mesa import Model, Agent
from mesa.space import NetworkGrid, SingleGrid
from mesa.time import RandomActivation
import networkx as nx
# from agents import Wappie

class Wappie(Agent):
    def __init__(self, unique_id: int, model: Model, pos) -> None:
        super().__init__(unique_id, model)
        self.opinion = self.random.random()
    
    def step(self):
        self.opinion = self.random.random()

class Political_spectrum(Model):
    def __init__(self, width: int, height: int) -> None:
        num_agents = width * height
        self.schedule = RandomActivation(self)

        # create (grid and) network
        self.G = nx.barabasi_albert_graph(n=num_agents, m=2)
        print(list(self.G.nodes))
        self.network = NetworkGrid(self.G)
        self.grid = SingleGrid(width, height, torus=True)

        # create agents
        for i in range(num_agents):
            x = i // width
            y = i % width
            agent = Wappie(i, self, (x, y))
            self.network.place_agent(agent, i)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True

    def step(self):
        self.schedule.step()

if __name__ == "__main__":
    model = Political_spectrum(3, 3)
