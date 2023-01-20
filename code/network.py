import networkx as nx
import random

from math import sqrt

def idealised_network(num_nodes):
    G = nx.Graph()

    # create temporary grid
    class Position():
        def __init__(self, id, pos):
            self.pos = pos
            self.id = id
            self.distances = []
        
        def add(self, other):
            distance = Distance(self, other)
            self.distances.append(distance)
            other.distances.append(distance)
        
        def dist(self, other):
            return sqrt((self.pos[0] - other.pos[0]) ** 2 + (self.pos[1] - other.pos[1]) ** 2)
        
        def closest(self, num: int):
            self.sort_dist()
            positions = []
            for distance in self.distances[:num]:
                positions.append(distance.other_pos(self))
            return positions
        
        def sort_dist(self):
            self.distances.sort()
    
    class Distance():
        def __init__(self, pos1, pos2) -> None:
            self.distance = pos1.dist(pos2)
            self.connects = (pos1, pos2)

        def __lt__(self, other):
            return self.distance < other.distance
        
        def other_pos(self, pos):
            for position in self.connects:
                if position != pos:
                    return position
            ValueError("This position is not connected to this node.")
    
    positions = []
    for i in range(num_nodes):
        network_position = (random.random(), random.random())
        pos = Position(i, network_position)
        for other_pos in positions:
            pos.add(other_pos)
        positions.append(pos)
        G.add_node(pos.id)
    for pos in positions:
        closest = pos.closest(10)
        for other_pos in closest:
            if len(G[other_pos.id]) < 5:
                G.add_edge(pos.id, other_pos.id)
            if len(G[pos.id]) == 5:
                break
    return G