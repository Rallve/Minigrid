from __future__ import annotations
import math
from operator import add
import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall, SpikeFloor, WorldObj
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.wrappers import DeadlySpikes


class FlowingLava(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=(2, 7),
        agent_start_dir=0,
        prob=0.2,
        max_steps: int | None = None,
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

        self.grid_memory = [[None for _ in range(self.width)] for _ in range(self.height)]

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
    
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 3, height - 2)

        # Wall in the middle
        for i in range(4):
            self.grid.set(4, 4 + i, Wall())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Place obstacles
        self.obstacles = []

        self.mission = "grand mission"

    def _move_obj(
        self,
        obj: WorldObj,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Move an object to the right
        """

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                obj.cur_pos[0] + 1,
                obj.cur_pos[1],
            )

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos
    
    def _update_grid_memory(self, pos):
        cell = self.grid.get(*pos)
        if self.grid_memory[pos[1]][pos[0]] is None:
            if cell is not None:
                self.grid_memory[pos[1]][pos[0]] = cell.type
            else:
                self.grid_memory[pos[1]][pos[0]] = "none"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        delete_list = []

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            new_pos = (old_pos[0] + 1, old_pos[1])

            if new_pos[0] < self.width:
                self._update_grid_memory(new_pos)

            if old_pos[0] == self.width - 1:
                delete_list.append(i_obst)
            else:
                try:
                    self._move_obj(
                        self.obstacles[i_obst], max_tries=100
                    )
                    self.grid.set(old_pos[0], old_pos[1], None)
                except Exception:
                    print("failed to place obstacle")
                    pass
            
            if self.grid_memory[old_pos[1]][old_pos[0]] != "none":
                self.grid.set(old_pos[0], old_pos[1], Wall())
        
        for i_obst in range(len(delete_list) - 1, -1, -1):
            self.obstacles.pop(delete_list[i_obst])

        # Place new obstacles
        if random.random() < self.prob:
            self.obstacles.append(Lava())
            self.put_obj(self.obstacles[-1], 0, 2)
            self._update_grid_memory((0, 2))
        
        if random.random() < self.prob:
            self.obstacles.append(Lava())
            self.put_obj(self.obstacles[-1], 0, 4)
            self._update_grid_memory((0, 4))
        
        if random.random() < self.prob:
            self.obstacles.append(Lava())
            self.put_obj(self.obstacles[-1], 0, 6)
            self._update_grid_memory((0, 6))

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Terminate the episode if the agent touches lava
        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell and agent_cell.type == "lava":
            terminated = True

        return obs, reward, terminated, truncated, info


def main():
    env = FlowingLava(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

    
if __name__ == "__main__":
    main()