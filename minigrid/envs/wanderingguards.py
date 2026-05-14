from __future__ import annotations
import math
from operator import add
import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Point, Wall, WorldObj
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class WanderingGuards(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 7),
        agent_start_dir=0,
        prob=0.5,
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

    def _place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        top = (max(top[0], 0), max(top[1], 0))

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):

            if random.random() < self.prob:

                old_pos = self.obstacles[i_obst].cur_pos
                top = tuple(map(add, old_pos, (-1, -1)))

                try:
                    self._place_obj(
                        self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                    )
                    self.grid.set(old_pos[0], old_pos[1], None)
                except Exception:
                    pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # Terminate the episode if the agent touches lava
        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell and agent_cell.type == "lava":
            terminated = True

        return obs, reward, terminated, truncated, info


def main():
    env = WanderingGuards(render_mode="human", prob=1)

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

    
if __name__ == "__main__":
    main()