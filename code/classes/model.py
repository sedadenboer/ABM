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
        grid_preference: float = None,
        grid_radius: int = 2,
        grid_density: float = 0.95,
        both_affected: bool = True
        ) -> None:
        """A model that simulates the change in belief based on local
        interactions and interactions on social media.

        Args:
            width (int, optional): The width of the grid. This together with
                grid_density determines the amount of agents. Defaults to 20.
            lambd (float, optional): Lambda, determines the distance in belief
                moved when two agents move away from each other.
                Defaults to 0.5.
            mu (float, optional): Mu, determines the distance in belief moved
                when two agents move towards each other. Defaults to 0.20.
            d1 (float, optional): The distance within which two agents will
                assimilate. Between 0 and the square root of two. Defaults to
                0.7.
            d2 (float, optional): The distance from which two agents will
                contrast. Should be larger than d1, but up to the square root
                of 2. Defaults to 1.0.
            satisfaction_distance (float, optional): The distance in belief for
                which agents are satisfied with their neighbours. Defaults to
                0.5.
            satisfaction_threshold (float, optional): The threshold in
                satisfaction. If the satisfaction score is lower, agents will
                be unhappy. Should be between -1 and 1. Defaults to 0.0.
            mu_norm (float, optional): The mean of the initial belief
                distribution. Defaults to 0.5.
            sigma_norm (float, optional): The standard deviation of the initial
                belief distribution. Defaults to 0.2.
            network_type (int | str, optional): The network used to simulate
                social media. Can be "BA", "idealised", "erdos-renyi" and
                "complete". Defaults to "BA".
            grid_preference (float, optional): When a float is provided, this
                chance will be used to determine whether an agent will connect
                with someone from the grid opposed to someone from the network.
                Defaults to None.
            grid_radius (int, optional): The radius for which agents in the
                grid are considered neighbours. Defaults to 2.
            grid_density (float, optional): The density of agents in the grid.
             Defaults to 0.95.
            both_affected (bool, optional): Whether both agents are affected by
                each interaction. Defaults to True.

        Raises:
            ValueError: When a network type is provided, that is not in the
                available network types.
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
        model_reporters = {"polarization": lambda m: m.polarization()}
        self.datacollector = DataCollector(model_reporters=model_reporters)

        # turn the model on
        self.running = True
        self.num_steps = 0
        self.datacollector.collect(self)

    def place_agent(self, unique_id: int, mu_norm: float, sigma_norm: float):
        """Creates an agent and places them on the grid and in the network.

        Args:
            unique_id (int): The unique id of the agent.
            mu_norm (float): The mean for the initial belief of the agent.
            sigma_norm (float): The standard deviation for the initial belief
                of the agent.
        """
        grid_pos = self.grid.find_empty()

        # create prior beliefs
        prog_cons = np.random.normal(mu_norm, sigma_norm)
        left_right = np.random.normal(mu_norm, sigma_norm)
        prior_beliefs = np.array([prog_cons, left_right])

        agent = Wappie(unique_id=unique_id, 
                        model=self,
                        grid_pos=grid_pos,
                        prior_beliefs=prior_beliefs)

        # place the agent in the grid and in the network
        self.grid.place_agent(agent, grid_pos)
        self.network.place_agent(agent, unique_id)

        # add the agent to the schedule
        self.schedule.add(agent)

        # save the agent in the model as well
        self.agents[unique_id] = agent

    def polarization(self):
        """Calculates the polarization in the model.
        Source:
        Koudenburg N, Kiers HAL, Kashima Y. A New Opinion Polarization Index
        Developed by Integrating Expert Judgments. Front Psychol.
        2021 Oct 13;12:738258. doi: 10.3389/fpsyg.2021.738258. PMID: 34721211;
        PMCID: PMC8549827.

        Returns:
            float: The polarisation in the model.
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

        if len(polarization) == 0:
            print(self.agents, self.num_agents) 
            raise ZeroDivisionError

        return sum(polarization) / len(polarization)

    def step(self):
        """Performs one time step of the model.
        """
        self.schedule.step()
        self.num_steps += 1
        self.datacollector.collect(self)

if __name__ == "__main__":
    # set parameters for Gaussian distribution
    mu = 0.5
    sigma = np.sqrt(0.2)

    # initialise model
    model = Political_spectrum(
        width=10,
        lambd=0.5,
        mu=0.20,
        d1=0.35,
        d2=1.0,
        mu_norm=mu,
        sigma_norm=sigma,
        network_type="complete",
        grid_radius=2,
        both_affected=True,
        grid_density=0.95
    )

    for _ in range(50):
        model.step()
