from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, SpikeFloor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SpikeCrossing(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        max_steps: int | None = None,
        prob=0.1,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.prob = prob

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        for column in range(1, 6):
            for row in range(2, 6):
                self.grid.set(column, row, SpikeFloor())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 4, 1)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def step(self, action):
        for cell in self.unwrapped.grid.grid:
            if cell is not None and cell.type == "spikefloor":
                cell.is_extended = self.np_random.choice([True, False], p=[self.prob, 1 - self.prob])
        
        obs, reward, terminated, truncated, info = super().step(action)

        current_cell = self.unwrapped.grid.get(*self.unwrapped.agent_pos)
        on_spike = current_cell is not None and current_cell.type == "spikefloor" and current_cell.is_extended

        if on_spike:
            terminated = True

        return obs, reward, terminated, truncated, info


def main():
    env = SpikeCrossing(render_mode="human", prob=0.1)

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

    
if __name__ == "__main__":
    main()