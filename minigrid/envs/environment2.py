from __future__ import annotations
from operator import add

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall, SpikeFloor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.wrappers import DeadlySpikes

# Two potential issues:
# 1. Obstacles may avoid the agent
# 2. Walking into a wall may terminate the episode
class Environment2(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 7),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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
    
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Block the goal with a locked door
        self.grid.set(width - 3, height - 2, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(width - 2, height - 3, Wall())
        self.grid.set(width - 3, height - 3, Wall())

        self.grid.set(4, 1, Key(COLOR_NAMES[0]))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Place obstacles
        self.obstacles = []
        
        self.obstacles.append(Lava())
        self.put_obj(self.obstacles[0], 1, 1)

        self.obstacles.append(Lava())
        self.put_obj(self.obstacles[1], self.width - 2, 1)

        self.mission = "grand mission"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and (front_cell.type != "goal" and front_cell.type != "key" and front_cell.type != "door")

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj( # IMPORTANT: obstacles may avoid the agent! FIX THIS LATER
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            #reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info


def main():
    env = Environment2(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

    
if __name__ == "__main__":
    main()