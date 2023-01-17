from mesa import Agent

class Individual(Agent):
    def __init__(self, unique_id: int, model: "Political_model", pos: tuple,
        	     starting_opinion: str
                 ) -> None:
        """Create an individual for the political model

        Args:
            unique_id (int): The unique id of the agent
            model (Political_model): The model of which this agent is a part.
            pos (tuple): The position of this agent.
            starting_opinion (str): The starting opinion of this agent.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.opinion = starting_opinion

    def step(self):
        """Performs one step of the agent. The agents looks at its neighbors
        and updates its opinion accordingly.
        """
        # get neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=self.model.radius)
        
        # find the number of people with opinion A and B
        num_A = 0
        num_B = 0
        for neighbor in neighbors:
            if neighbor.opinion == "A":
                num_A += 1
            else:
                num_B += 1
        total = num_A + num_B

        # chance of changing opinion
        chance_to_B = (num_B / total) ** self.model.alpha_AB
        chance_to_A = (num_A / total) ** self.model.alpha_BA

        # change opinion
        if self.opinion == "A" and self.random.random() < chance_to_B:
            self.opinion = "B"
            self.model.num_A -= 1
            self.model.num_B += 1
        elif self.opinion == "B" and self.random.random() < chance_to_A:
            self.opinion = "A"
            self.model.num_B -= 1
            self.model.num_A += 1
