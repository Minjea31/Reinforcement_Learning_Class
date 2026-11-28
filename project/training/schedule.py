"""A bounded transition between two fixed curriculum stages."""
from dataclasses import asdict,dataclass


@dataclass
class StageHandoff:
    from_stage:int
    at_steps:int
    to_stage:int
    additional_steps:int
    done:bool=False

    def __post_init__(self):
        if self.from_stage not in (1,2,3) or self.to_stage!=self.from_stage+1:
            raise ValueError('A handoff must advance exactly one curriculum stage')
        if self.at_steps<=0 or self.additional_steps<=0:
            raise ValueError('Stage budgets must be positive')

    def due(self,stage,steps):
        return not self.done and stage==self.from_stage and steps>=self.at_steps

    @property
    def final_steps(self):return self.at_steps+self.additional_steps

    def state_dict(self):return asdict(self)
