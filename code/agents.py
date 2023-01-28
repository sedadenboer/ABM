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
        beliefs_copy += self.model.mu * (self.beliefs - other.beliefs)
        self.beliefs = beliefs_copy
        
        if self.model.both_affected:
            neighbor_beliefs_copy += self.model.mu * (other.beliefs - self.beliefs)
            other.beliefs = neighbor_beliefs_copy

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

    def step(self):
        # get neighbors from grid and network
        neighbors_grid = self.model.grid.get_neighbors(self.grid_pos, moore=True, radius=self.model.grid_radius)
        grid_chances = [self.model.p_grid/len(neighbors_grid) for _ in neighbors_grid]

        connected_nodes = self.model.network.get_neighbors(self.unique_id)
        neighbors_network = [self.model.agents[i] for i in connected_nodes]
        network_chances = [self.model.p_network/len(neighbors_network) for _ in neighbors_network]

        # merge neighbors
        connections = neighbors_grid + neighbors_network
        weights = grid_chances + network_chances

        # choose random connection to interact with
        if len(connections) == 0:
            return
        interacting_neighbor = self.random.choices(connections, weights)[0]
        # print("other:" + str(interacting_neighbor))

        if self.distance(interacting_neighbor) < self.model.d1:
            self.assimilation(interacting_neighbor)
        elif self.distance(interacting_neighbor) > self.model.d2:
            self.contrast(interacting_neighbor)
