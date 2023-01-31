from mesa import Model, Agent
from mesa.space import NetworkGrid, SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np

from agents import Wappie

class Political_spectrum(Model):

    network_types = ["BA", "idealised", "erdos-renyi", "complete"]

    def __init__(
        self,
        width: int = 20,
        lambd: float = 0.5,
        mu: float = 0.20,
        d1: float = 0.7,
        d2: float = 1.0,
        satisfaction_distance: float = 0.5,
        satisfaction_threshold: float = 0.0,
        mu_norm: float = 0.5,
        sigma_norm: float = 0.2,
        network_type: str = "BA",
        grid_preference: float = 0.5,
        grid_radius: int = 2,
        grid_density: float = 1,
        both_affected: bool = True
        ) -> None:
        """A model for people changing opinion on their political beliefs.

        Args:
            width (int): The width of the grid used.
            height (int): The height of the grid used.
            network_type (str): Can be BA, random, idealised or scale_free?
        """
        height = width
        self.num_agents = round(grid_density * width * height)
        self.schedule = RandomActivation(self)

        # create network
        if type(network_type) == int:
            network_type = self.network_types[network_type]

        if network_type == "BA":
            self.G = nx.barabasi_albert_graph(n=self.num_agents, m=2)
        elif network_type == "idealised":
            self.G = nx.random_geometric_graph(self.num_agents, 0.15)
        elif network_type == "erdos-renyi":
            self.G = nx.erdos_renyi_graph(self.num_agents, 0.4)
        elif network_type == "complete":
            self.G = nx.complete_graph(self.num_agents)
        else:
            raise ValueError("'network_type' should be either BA, erdos-renyi, complete or idealised")

        self.network = NetworkGrid(self.G)

        self.grid = SingleGrid(width, height, torus=True)
        self.grid_radius = grid_radius

        # create agents
        self.agents = {}
        for i in range(self.num_agents):
            self.place_agent(i, mu_norm, sigma_norm)

        # save the parameters
        self.d1 = d1
        self.d2 = d2
        self.ds = satisfaction_distance
        self.threshold = satisfaction_threshold
        self.lambd = lambd
        self.mu = mu
        self.p_grid = grid_preference
        # self.p_network = 1 - grid_preference
        self.both_affected = both_affected

        # set the datacollector
        model_reporters = {"polarization": lambda m: m.polarization(),
                        "network_influence": lambda m: m.influenced_by_network()}
                        # "step": lambda m: m.num_steps}
        self.datacollector = DataCollector(model_reporters=model_reporters)

        # turn the model on
        self.running = True
        self.num_steps = 0
        self.datacollector.collect(self)

    def place_agent(self, unique_id, mu_norm, sigma_norm):
        grid_pos = self.grid.find_empty()

        # create prior beliefs
        prog_cons = np.random.normal(mu_norm, sigma_norm)
        left_right = np.random.normal(mu_norm, sigma_norm)
        prior_beliefs = np.array([prog_cons, left_right])

        agent = Wappie(unique_id=unique_id, 
                        model=self,
                        grid_pos=grid_pos,
                        prior_beliefs=prior_beliefs)

        self.grid.place_agent(agent, grid_pos)
        # print(agent.pos)
            
        self.network.place_agent(agent, unique_id)
        # print(agent.pos)

        self.schedule.add(agent)

        self.agents[unique_id] = agent

    def polarization(self):
        """Calculates the polarization in the model.
        Source:
        Koudenburg N, Kiers HAL, Kashima Y. A New Opinion Polarization Index
        Developed by Integrating Expert Judgments. Front Psychol.
        2021 Oct 13;12:738258. doi: 10.3389/fpsyg.2021.738258. PMID: 34721211;
        PMCID: PMC8549827.

        Returns:
            _type_: _description_
        """
        # only measure every 100 steps
        if self.num_steps % 100 != 0:
            return
        
        polarization = []
        for agent1_index in range(self.num_agents):
            for agent2_index in range(agent1_index + 1, self.num_agents):
                agent1 = self.agents[agent1_index]
                agent2 = self.agents[agent2_index]
                # find the distance between the agents
                dist = agent1.distance(agent2)
                polarization.append(dist)
        return sum(polarization) / len(polarization)
        # return sum(polarization)

    def influenced_by_network(self):
        if self.num_steps % 100 != 0:
            return
            
        influences = []
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            if agent.distance_in_network() == 0:
                influenced = 0
            else:
                influenced = agent.distance_in_grid() / agent.distance_in_network()
            influences.append(influenced)
        return sum(influences) / self.num_agents

    def step(self):
        self.schedule.step()
        self.num_steps += 1
        self.datacollector.collect(self)

if __name__ == "__main__":
    # model = Political_spectrum(10, 10, "scale_free")
    # nx.draw_kamada_kawai(model.G, node_size=50)
    # plt.savefig("../kamada_kawai.png")

    # set parameters for Gaussian distribution
    mu = 0.5
    sigma = np.sqrt(0.2)

    # initialise model
    model = Political_spectrum(
        width=25,
        lambd=0.05,
        mu=0.20,
        d1=0.35,
        d2=1.5,
        mu_norm=mu,
        sigma_norm=sigma,
        network_type="erdos-renyi",
        grid_preference=0.5,
        grid_radius=2,
        both_affected=True
    )

    for _ in range(100):
        model.step()
