from mesa import Agent, Model
from typing import List
from scipy.spatial import distance
import numpy as np
import copy


class Wappie(Agent):
    def __init__(self, unique_id: int, model: Model, grid_pos, prior_beliefs) -> None:
        super().__init__(unique_id, model)

        self.beliefs = prior_beliefs
        self.normalize_beliefs()
        self.interacting_neighbor = None
        self.unique_id = unique_id
        self.grid_pos = grid_pos
        # print(self.pos)

    def distance(self, other):
        """ """
        # print(self.beliefs, other.beliefs)
        return distance.euclidean(list(self.beliefs), list(other.beliefs))

    def assimilation(self, other):
        """ """
        # make a copy of beliefs
        beliefs_copy = copy.copy(self.beliefs)
        neighbor_beliefs_copy = copy.copy(other.beliefs)

        # update beliefs of agent and interacting connection
        self.beliefs += self.model.mu * (neighbor_beliefs_copy - beliefs_copy)
        if self.model.both_affected:
            other.beliefs += self.model.mu * (beliefs_copy - neighbor_beliefs_copy)

    def normalisation(self, x):
        """"""
        if x < 0.0:
            return 0.0  # should be float, otherwise numpy will scream at you...
        elif x > 1.0:
            return 1.0
        else:
            return x

    def normalize_beliefs(self):
        for i in range(len(self.beliefs)):
            if self.beliefs[i] < 0.0:
                self.beliefs[i] = 0.0
            elif self.beliefs[i] > 1.0:
                self.beliefs[i] = 1.0

    def contrast(self, other):
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
        # calculate the average distance in beliefs and that of grid neighbours
        distances = []
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=self.model.grid_radius)
        for neighbor in neighbors_grid:
            distances.append(self.distance(neighbor))
        return sum(distances) / len(distances)
    
    def distance_in_network(self):
        connected_nodes = self.model.network.get_neighbors(self.unique_id)
        if len(connected_nodes) == 0:
            return 0
        neighbors_network = [self.model.agents[i] for i in connected_nodes]
        distances = []
        for neighbor in neighbors_network:
            distances.append(self.distance(neighbor))
        return sum(distances) / len(distances)
    
    def interact(self):
        # get neighbors from grid and network
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=self.model.grid_radius)

        connected_nodes = self.model.network.get_neighbors(self.unique_id)
        neighbors_network = [self.model.agents[i] for i in connected_nodes]

        # choose neighbors
        if np.random.random() < self.model.p_grid:
            if not neighbors_grid:
                return
            interacting_neighbor = self.random.choice(neighbors_grid)
        else:
            if not neighbors_network:
                return
            interacting_neighbor = self.random.choice(neighbors_network)
        
        # print("other:" + str(interacting_neighbor))

        if self.distance(interacting_neighbor) < self.model.d1:
            self.assimilation(interacting_neighbor)
        elif self.distance(interacting_neighbor) > self.model.d2:
            self.contrast(interacting_neighbor)
    
    def satisfied(self):
        """Arcón, Victoria, Juan Pablo Pinasco, and Inés Caridi.
        "A Schelling-Opinion Model Based on Integration of Opinion Formation
        with Residential Segregation." Causes and Symptoms of Socio-Cultural
        Polarization: Role of Information and Communication Technologies.
        Singapore: Springer Singapore, 2022. 27-50.

        Returns:
            _type_: _description_
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
        
        return satisfaction / len(neighbors_grid) > self.model.threshold

    def move(self):
        # find the position that has the most neighbours like itself
        new_pos = self.model.grid.find_empty()
        if new_pos:
            print(new_pos)
            self.pos = self.grid_pos
            self.model.grid.move_agent(self, new_pos)
            self.grid_pos = new_pos

    def step(self):
        self.interact()
        if not self.satisfied():
            self.move()
