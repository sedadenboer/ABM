from mesa import Agent

class Wappie(Agent):
    def __init__(
        self, unique_id: int, model: Model, pos, prior_beliefs: List[float]
    ) -> None:
        super().__init__(unique_id, model)

        self.beliefs = prior_beliefs
        self.interacting_neighbor = None
        self.unique_id = unique_id

    def distance(self):
        return distance.euclidean(self.beliefs, self.interacting_neighbor.beliefs)

    def assimilation(self):
        """ """
        beliefs_copy = copy.copy(self.beliefs)
        interacting_neighbor_beliefs_copy = copy.copy(self.interacting_neighbor.beliefs)

        beliefs_copy += self.model.mu * np.subtract(
            beliefs_copy, interacting_neighbor_beliefs_copy
        )
        interacting_neighbor_beliefs_copy += self.model.mu * np.subtract(
            interacting_neighbor_beliefs_copy, beliefs_copy
        )

        self.beliefs = beliefs_copy
        self.interacting_neighbor.beliefs = interacting_neighbor_beliefs_copy

    def normalisation(self, x):
        if x < 0:
            return 0
        elif x > 1:
            return 1
        else:
            return x

    def contrast(self):

        prog_cons_1 = self.beliefs[0]
        left_right_1 = self.beliefs[1]

        prog_cons_2 = self.interacting_neighbor[0]
        left_right_2 = self.interacting_neighbor[1]

        new_prog_cons_1 = self.normalisation(
            prog_cons_1 - self.model.lambd * (prog_cons_2 - prog_cons_1)
        )
        new_prog_cons_2 = self.normalisation(
            prog_cons_2 - self.model.lambd * (prog_cons_1 - prog_cons_2)
        )

        new_left_right_1 = self.normalisation(
            left_right_1 - self.model.lambd * (left_right_2 - left_right_1)
        )
        new_left_right_2 = self.normalisation(
            left_right_2 - self.model.lambd * (left_right_1 - left_right_2)
        )

        self.belief = np.array([new_prog_cons_1, new_left_right_1])
        self.interacting_neighbor.beliefs = np.array(
            [new_prog_cons_2, new_left_right_2]
        )

    def step(self):
        # get neighbors from grid and network
        neighbors_grid = self.model.grid.get_neighbors(self.pos, moore=True)
        neighbors_network = self.model.network.get_neighbors(self.unique_id, moore=True)

        # merge neighbors
        connections = neighbors_grid + neighbors_network

        # choose random connection to interact with
        if len(connections) > 0:
            self.interacting_neighbor = random.choice(connections)

        if self.distance() < self.model.d1:
            self.assimilation()
        elif self.distance() > self.model.d2:
            self.contrast()