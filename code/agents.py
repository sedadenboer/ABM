from mesa import Agent

class Individual(Agent):
    def __init__(self, unique_id: int, model: "Political_model", pos: tuple,
        	     starting_opinion: str
                 ) -> None:
        super().__init__(unique_id, model)
        self.pos = pos
        self.opinion = starting_opinion

    def step(self):
        # get neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=self.model.radius)
        
        num_A = 0
        num_B = 0
        for neighbor in neighbors:
            if neighbor.opinion == "A":
                num_A += 1
            else:
                num_B += 1
        total = num_A + num_B

        # change opinion
        chance_to_B = (num_B / total) ** self.model.alpha_AB
        chance_to_A = (num_A / total) ** self.model.alpha_BA

        if self.opinion == "A" and self.random.random() < chance_to_B:
            self.opinion = "B"
            self.model.num_A -= 1
            self.model.num_B += 1
        elif self.opinion == "B" and self.random.random() < chance_to_A:
            self.opinion = "A"
            self.model.num_B -= 1
            self.model.num_A += 1
