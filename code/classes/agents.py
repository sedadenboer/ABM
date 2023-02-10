# agents.py
#
# Course: ABM (2022/2023)
#
# Description: Contains the class for an agent
# in the political spectrum model.

from mesa import Agent, Model
from scipy.spatial import distance
import numpy as np
import copy


class Wappie(Agent):
    def __init__(self, 
                 unique_id: int,
                 model: Model,
                 grid_pos: tuple,
                 prior_beliefs: tuple) -> None:
        """The agents of the Policital_Spectrum model.
        Has a position in both the grid and the network of the model.
        Changes belief based on its connections on the grid and in the network.

        Args:
            unique_id (int): The id of the agent.
            model (Model): The model this agent belongs to.
            grid_pos (tuple[int]): The position of this agent in the grid.
            prior_beliefs (tuple[float]): The initial belief of this agent.
                Consist of two floats corresponding to the x and y position in
                the belief space.
        """
        super().__init__(unique_id, model)

        self.beliefs = prior_beliefs
        self.normalize_beliefs()
        self.interacting_neighbor = None
        self.unique_id = unique_id
        self.grid_pos = grid_pos
        self.influenced_by_grid = 0
        self.influenced_by_network = 0
        self._tracker = "grid"

    def set_influence_tracker(self, to_change: str):
        assert to_change == "grid" or to_change == "network"
        self._tracker = to_change

    @property
    def influence_tracker(self):
        if self._tracker == "grid":
            return self.influenced_by_grid
        return self.influenced_by_network
    
    @influence_tracker.setter
    def influence_tracker(self, value):
        if self._tracker == "grid":
            self.influenced_by_grid = value
        self.influenced_by_network = value

    def distance(self, other):
        """Determine the distance in belief between this agent and another.
        Uses eudclidean distance

        Args:
            other (Wappie): The agent to determine the distance to.

        Returns:
            float: The distance between the two agents.
        """
        return distance.euclidean(list(self.beliefs), list(other.beliefs))

    def normalisation(self, x):
        """Makes sure the coordinate stays within the boundaries of the space.
        Any coordinate smaller than 0 or larger than 1 will be normalized to
        respectively 0 or 1.

        Args:
            x (float): The coordinate to be normalized.

        Returns:
            float: The normalized value of the coordinate.
        """
        if x < 0.0:
            return 0.0 
        elif x > 1.0:
            return 1.0
        else:
            return x

    def normalize_beliefs(self):
        """Normalize the belief within the given space.
        """
        for i in range(len(self.beliefs)):
            self.beliefs[i] = self.normalisation(self.beliefs[i])

    def assimilation(self, other: "Wappie"):
        """Move the beliefs of two agents towards each other.

        Args:
            other (Wappie): The agent to which the belief is moved.
        """
        # make a copy of beliefs
        beliefs_copy = copy.copy(self.beliefs)
        neighbor_beliefs_copy = copy.copy(other.beliefs)

        # update beliefs of agent and interacting connection
        self.beliefs += self.model.mu * (neighbor_beliefs_copy - beliefs_copy)
        if self.model.both_affected:
            other.beliefs += self.model.mu * (beliefs_copy - neighbor_beliefs_copy)

    def contrast(self, other: "Wappie"):
        """Move two agents beliefs away from each other.

        Args:
            other (Wappie): The agent from which to move away.
        """
        # make a copy of beliefs
        beliefs_copy = copy.copy(self.beliefs)
        neighbor_beliefs_copy = copy.copy(other.beliefs)

        # vectorize normalisation vector s.t. it can be applied to an numpy array
        add_normalisation = np.vectorize(self.normalisation)

        # update beliefs of agent and interacting connection
        beliefs_copy = add_normalisation(
            self.beliefs - self.model.lambd * (other.beliefs - self.beliefs)
            )
        self.beliefs = beliefs_copy
        self.normalize_beliefs()

        if self.model.both_affected:
            neighbor_beliefs_copy = add_normalisation(
                other.beliefs - self.model.lambd * (self.beliefs - other.beliefs)
                )
            other.beliefs = neighbor_beliefs_copy
            other.normalize_beliefs()

    def distance_in_grid(self):
        """Gives the average distance in belief to all grid neighbours.

        Returns:
            float: The average distance.
        """
        # calculate the average distance in beliefs and that of grid neighbours
        distances = []
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=self.model.grid_radius)
        for neighbor in neighbors_grid:
            distances.append(self.distance(neighbor))
        return sum(distances) / len(distances)
    
    def distance_in_network(self):
        """Gives the average distance in belief to all network neighbours.

        Returns:
            float: The average distance.
        """
        connected_nodes = self.model.network.get_neighbors(self.unique_id)
        if len(connected_nodes) == 0:
            return 0
        neighbors_network = [self.model.agents[i] for i in connected_nodes]
        distances = []
        for neighbor in neighbors_network:
            distances.append(self.distance(neighbor))
        return sum(distances) / len(distances)
    
    def interact(self):
        """Interact with a neighbour in either the grid or the network.
        """
        # get neighbors from grid and network
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=self.model.grid_radius)
        
        connected_nodes = self.model.network.get_neighbors(self.unique_id)
        neighbors_network = [self.model.agents[i] for i in connected_nodes]

        # choose neighbors
        if self.model.p_grid == None:
            # print(neighbors_grid + neighbors_network)
            interacting_neighbor = self.random.choice(neighbors_grid + neighbors_network)
            if interacting_neighbor in neighbors_grid:
                self.set_influence_tracker("grid")
            else:
                self.set_influence_tracker("network")
        elif np.random.random() < self.model.p_grid:
            if not neighbors_grid:
                return
            interacting_neighbor = self.random.choice(neighbors_grid)
            self.set_influence_tracker("grid")
        else:
            if not neighbors_network:
                return
            self.set_influence_tracker("network")
            interacting_neighbor = self.random.choice(neighbors_network)

        if self.distance(interacting_neighbor) < self.model.d1:
            self.assimilation(interacting_neighbor)
            self.influence_tracker += 1
        elif self.distance(interacting_neighbor) > self.model.d2:
            self.contrast(interacting_neighbor)
            self.influence_tracker += 1
    
    def satisfied(self):
        """
        Checks whether this agent is satisfied with the neighbouring agents in the grid.
        Source: Arcón, Victoria, Juan Pablo Pinasco, and Inés Caridi.
        "A Schelling-Opinion Model Based on Integration of Opinion Formation
        with Residential Segregation." Causes and Symptoms of Socio-Cultural
        Polarization: Role of Information and Communication Technologies.
        Singapore: Springer Singapore, 2022. 27-50.

        Returns:
            bool: If the agent is satisfied with their neighbours or not.
        """
        # get grid neighbors
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=1)

        # is this agent satisfied?
        satisfaction = 0
        for neighbor in neighbors_grid:
            if self.distance(neighbor) < self.model.ds:
                # satisfied
                satisfaction += 1
            else:
                satisfaction -= 1
        
        # avoid errors
        if len(neighbors_grid) == 0:
            return 1

        return satisfaction / len(neighbors_grid) > self.model.threshold

    def move(self):
        """Move the agent to a random empty spot if there is one.
        """
        # find the position that has the most neighbours like itself
        new_pos = self.model.grid.find_empty()
        if new_pos:
            self.pos = self.grid_pos
            self.model.grid.move_agent(self, new_pos)
            self.grid_pos = new_pos

    def step(self):
        """Perform one time step for this agent.
        """
        self.interact()
        if not self.satisfied():
            self.move()
