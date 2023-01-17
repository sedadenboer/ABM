from mesa import Agent

class Wappie(Agent):
    def __init__(self, unique_id: int, model: "Model") -> None:
        super().__init__(unique_id, model)
    
    def step(self):
        pass