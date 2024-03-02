from collections import OrderedDict
import gym
from gym import spaces
import numpy as np

from typing import List, Tuple, Optional, Dict, Iterable

import networkx as nx
from rware.entities import (
    _LAYER_AGENTS,
    _LAYER_SHELFS,
    _LAYER_LOADERS,
    _COLLISION_LAYERS,
    VectorWriter,
    Action,
    Direction,
    RewardType,
    ObserationType,
    ImageLayer,
    Agent,
    Shelf,
)

class Warehouse(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        agent_type: str = 'cl',
        layout: str = None,
        observation_type: ObserationType=ObserationType.FLATTENED,
        image_observation_layers: List[ImageLayer]=[
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE
        ],
        image_observation_directional: bool=True,
        normalised_coordinates: bool=False,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.n_agents = n_agents
        if agent_type is None:
            self.agent_type = (['c', 'l', 'cl']*self.n_agents)[:self.n_agents]
        elif isinstance(agent_type, Iterable) and not isinstance(agent_type, str):
            assert len(agent_type) == self.n_agents, "agent_type must be a scalar or a list (found {0}) of length n_agents (is {1}) ".format( len(agent_type) , self.n_agents)
            for type_ in agent_type:
                assert type_ in ('c', 'l', 'cl'), "invalid input: agent_type must be either c (carrying), l (loading) or cl (carrying or loading) but recived {0}".format(type_)
            self.agent_type = agent_type
        else:
            self.agent_type = [agent_type] * self.n_agents

        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.observation_space = None
        if observation_type == ObserationType.IMAGE:
            self._use_image_obs(image_observation_layers, image_observation_directional)
        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if observation_type == ObserationType.FLATTENED:
            self._use_fast_obs()

        self.renderer = None

        self._make_subtasks_mask()


    def _make_subtasks_mask(self):

        # make subtasks mask matrix for agents of the size n_agents x n_subtasks(3)
        # subtask 1: locating - go to a requested shelf
        # subtask 2: colaborating - check if a loader (for carrier agents) or a carrier (for loader agents) is in the same location
        # subtask 3: loading - load a requested shelf (either independently or colaboratively)
        # subtask 4: delivery - deliver a shelf to the goal location
        # subtask 5: removing subtask rewards when next subtask is done (to avoid incomplete task behaviours)

        self.subtasks_mask = np.zeros((self.n_agents, 5))
        for i,agent_type in enumerate(self.agent_type):
            if agent_type=='c':
                self.subtasks_mask[i,:] = [1, 1, 1, 1, 1]
            elif agent_type=='l':
                self.subtasks_mask[i,:] = [1, 1, 1, 0, 1]
            else:
                self.subtasks_mask[i,:] = [1, 0, 1, 1, 1]


    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (self.column_height + 1) == 0)  # horizontal highways
            or (y == self.grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > self.grid_size[0] - (self.column_height + 3))
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
        )
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION or layer == ImageLayer.LOADERS_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max([d.value + 1 for d in Direction])
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        self.observation_space = spaces.Tuple(
            tuple([spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_bits_for_self = 6 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent * 2 # multiplied by two to include loader agents as well
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        if self.normalised_coordinates:
            location_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
            )
        else:
            location_space = spaces.Dict(
                OrderedDict(
                    {
                        "x": spaces.Discrete(1),
                        "y": spaces.Discrete(1),
                    }
                )

            )
            # location_space = spaces.MultiDiscrete(
            #     [self.grid_size[1], self.grid_size[0]]
            # )

        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Dict(
                        OrderedDict(
                            {
                                "self": spaces.Dict(
                                    OrderedDict(
                                        {
                                            "location": location_space,
                                            "can_carry": spaces.MultiBinary(1),
                                            "can_load": spaces.MultiBinary(1),
                                            "has_located": spaces.MultiBinary(1),
                                            "has_collaborated" : spaces.MultiBinary(1),
                                            "has_loaded": spaces.MultiBinary(1),
                                            "has_delivered": spaces.MultiBinary(1),
                                            "carrying_shelf": spaces.MultiBinary(1),
                                            "direction": spaces.Discrete(4),
                                            "on_highway": spaces.MultiBinary(1),
                                        }
                                    )
                                ),
                                "sensors": spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (
                                        spaces.Dict(
                                            OrderedDict(
                                                {
                                                    "has_agent": spaces.MultiBinary(1),
                                                    "direction": spaces.Discrete(4),
                                                    "local_message": spaces.MultiBinary(
                                                        self.msg_bits
                                                    ),
                                                    "has_loader": spaces.MultiBinary(1),
                                                    "direction_loader": spaces.Discrete(4),
                                                    "local_message_loader": spaces.MultiBinary(
                                                        self.msg_bits
                                                    ),
                                                    "has_shelf": spaces.MultiBinary(1),
                                                    "shelf_requested": spaces.MultiBinary(
                                                        1
                                                    ),
                                                }
                                            )
                                        ),
                                    )
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_obs(self, agent):
        if self.image_obs:
            # write image observations
            if agent.id == 1:
                layers = []
                # first agent's observation --> update global observation layers
                for layer_type in self.image_observation_layers:
                    if layer_type == ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                        # print("SHELVES LAYER")
                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0
                        # print("REQUESTS LAYER")
                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                        # print("AGENT DIRECTIONS LAYER")
                    elif layer_type == ImageLayer.LOADERS:
                        layer = self.grid[_LAYER_LOADERS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.LOADERS_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_load and not ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                        # print("AGENT DIRECTIONS LAYER")
                    elif layer_type == ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.carrying_shelf is not None:
                                layer[ag.x, ag.y] = 1.0
                        # print("AGENT LOAD LAYER")
                    elif layer_type == ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal_y, goal_x in self.goals:
                            layer[goal_x, goal_y] = 1.0
                        # print("GOALS LAYER")
                    elif layer_type == ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            layer[ag.y, ag.x] = 0.0
                        # print("ACCESSIBLE LAYER")
                    # print(layer)
                    # print()
                    # pad with 0s for out-of-map cells
                    layer = np.pad(layer, self.sensor_range, mode="constant")
                    layers.append(layer)
                self.global_layers = np.stack(layers)

            # global information was generated --> get information for agent
            start_x = agent.y
            end_x = agent.y + 2 * self.sensor_range + 1
            start_y = agent.x
            end_y = agent.x + 2 * self.sensor_range + 1
            obs = self.global_layers[:, start_x:end_x, start_y:end_y]

            if self.image_observation_directional:
                # rotate image to be in direction of agent
                if agent.dir == Direction.DOWN:
                    # rotate by 180 degrees (clockwise)
                    obs = np.rot90(obs, k=2, axes=(1,2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1,2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1,2))
                # no rotation needed for UP direction
            return obs

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_loaders = np.pad(
                self.grid[_LAYER_LOADERS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
            padded_loaders = self.grid[_LAYER_LOADERS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        loaders = padded_loaders[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            obs = VectorWriter(self.observation_space[agent.id - 1].shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, 
                       int(agent.can_carry), 
                       int(agent.can_load), 
                       int(agent.has_located), 
                       int(agent.has_collaborated), 
                       int(agent.has_loaded), 
                       int(agent.has_delivered) ,
                       int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])

            for i, (id_agent, id_shelf, id_loader) in enumerate(zip(agents, shelfs, loaders)):
                if id_agent == 0:
                    obs.skip(1)
                    obs.write([1.0])
                    obs.skip(3 + self.msg_bits)
                else:
                    obs.write([1.0])
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)
                if id_loader == 0:
                    obs.skip(1)
                    obs.write([1.0])
                    obs.skip(3 + self.msg_bits)
                else:
                    obs.write([1.0])
                    direction = np.zeros(4)
                    direction[self.agents[id_loader - 1].dir.value] = 1.0
                    obs.write(direction)
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_loader - 1].message)
                if id_shelf == 0:
                    obs.skip(2)
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )

            return obs.vector
 
        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "can_carry": [int(agent.can_carry)],
            "can_load": [int(agent.can_load)],
            "has_located": [int(agent.has_located)],
            "has_collaborated": [int(agent.has_collaborated)],
            "has_loaded": [int(agent.has_loaded)],
            "has_delivered": [int(agent.has_delivered)],
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message

        # find neighboring agents
        for i, id_ in enumerate(loaders):
            if id_ == 0:
                obs["sensors"][i]["has_loader"] = [0]
                obs["sensors"][i]["direction_loader"] = 0
                obs["sensors"][i]["local_message_loader"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_loader"] = [1]
                obs["sensors"][i]["direction_loader"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message_loader"] = self.agents[id_ - 1].message

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            if s.id>0:
                self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            if a.can_load and not a.can_carry:
                self.grid[_LAYER_LOADERS, a.y, a.x] = a.id
            else:
                self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

        # refill request queue when a shelf is delivered
        if len(self.removed_shelf_ids) > 0:
            for s in self.shelfs:
                if s.id == 0:
                    empty_shelf_locations = self.find_empty_shelf_locations()
                    s.id = self.removed_shelf_ids.pop()
                    s.x, s.y = empty_shelf_locations.pop()
                    # add a new random shelf to request queue but not repeating the same shelf in the queue
                    eligible_shelfs = [shelf for shelf in self.shelfs if shelf not in self.request_queue]
                    self.request_queue.append(np.random.choice(eligible_shelfs))

    def find_empty_shelf_locations(self):
        find_empty_shelf_locations = []
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                if self._is_highway(x, y) == 0 and self.grid[_LAYER_SHELFS, y, x] == 0:
                    find_empty_shelf_locations.append((x,y))
        return find_empty_shelf_locations

    def _set_agent_types(self):
        for agent, type_ in zip(self.agents, self.agent_type):
            if type_=='c':
                agent.can_carry = True
                agent.can_load = False
            elif type_=='l':
                agent.can_carry = False
                agent.can_load = True
            else:
                agent.can_carry = True
                agent.can_load = True

    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]

        # spawn agents at random locations
        agent_locs = np.random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]

        self.removed_shelf_ids = []

        self._set_agent_types()

        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )

        return tuple([self._make_obs(agent) for agent in self.agents])
        # for s in self.shelfs:
        #     self.grid[0, s.y, s.x] = 1
        # print(self.grid[0])
    
    def resolve_move_conflict(self, agent_list, env_grid, all_agents):

        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        commited_agents = set()

        G = nx.DiGraph()

        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            if (
                agent.carrying_shelf
                and start != target
                and env_grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    env_grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents[
                        env_grid[_LAYER_AGENTS, target[1], target[0]] - 1
                    ].carrying_shelf
                )
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = env_grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = env_grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([all_agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents

        for agent in failed_agents:
            # print( agent.x, agent.y , ':' , agent.req_action)
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

    def find_nearest_loader(self, grid, row, col):
        # Define the boundaries of the 3x3 window
        start_row = max(row - 1, 0)
        end_row = min(row + 2, grid[_LAYER_LOADERS].shape[0])
        start_col = max(col - 1, 0)
        end_col = min(col + 2, grid[_LAYER_LOADERS].shape[1])
    
        # Iterate through the window and check for non-zero values
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                if grid[_LAYER_LOADERS,i, j] != 0:
                    return grid[_LAYER_LOADERS,i,j]
        return None
    
    def transition_function(self):

        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf and agent.can_carry:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                loader_id = self.grid[_LAYER_LOADERS, agent.y, agent.x]
                # loader_id = self.find_nearest_loader(self.grid, agent.y, agent.x )
                if shelf_id and (agent.can_load or loader_id):
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
                            
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                loader_id = self.grid[_LAYER_LOADERS, agent.y, agent.x]
                # loader_id = self.find_nearest_loader(self.grid, agent.y, agent.x )
                if not self._is_highway(agent.x, agent.y):
                    if agent.can_load:
                        agent.carrying_shelf = None
                        # print('unloaded on its own')
                    elif not agent.can_load and loader_id:
                        agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        # rewards[agent.id - 1] += 0.5
                        raise NotImplementedError('TWO_STAGE reward not implemenred for diverse rware')

            self._recalc_grid()

    def reward_function(self):

        # create reward subtask array
        # subtask 1: locating - go to a requested shelf
        # subtask 2: colaborating - check if a loader (for carrier agents) or a carrier (for loader agents) is in the same location
        # subtask 3: loading - load a requested shelf (either independently or colaboratively)
        # subtask 4: delivery - deliver a shelf to the goal location
        # subtask 5: removing subtask rewards when next subtask is done (to avoid incomplete task behaviours)
        #
        # reward array:
        #           [subtask 1, subtask 2, subtask 3, subtask 4, subtask 5]
        # agent c:  [      1/7,       3/7,       5/7,         1,        0]     x 1/2
        # agent l:  [      1/5,       3/5,       5/5,         0,        0]     x 1/2
        # agent cl: [      1/5,        0,        3/5,       5/5,        0]
        # :                  :         :         :         :         :
         
        reward_array = np.zeros_like(self.subtasks_mask)

        for agent in self.agents:

            # subtask-reward allocation
            
            # if carrying agent has delivered in previous step, turn all subtask flag to false
            if agent.has_delivered:
                agent.has_located = False
                agent.has_collaborated = False
                agent.has_loaded = False
                agent.has_delivered = False

            # if loading agent has loaded in previous step, turn all subtask flag to false
            if agent.has_loaded:
                agent.has_located = False
                agent.has_collaborated = False
                agent.has_loaded = False
                
            # subtask 1: locating - go to a requested shelf
            if not agent.has_located and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if self.shelfs[shelf_id-1] in self.request_queue:
                    if agent.can_carry and not agent.can_load:
                        reward_array[agent.id - 1, 0] = 1/7
                    else:
                        reward_array[agent.id - 1, 0] = 1/5
                    agent.has_located = True
                    print('located shelf')
            
            # subtask 2: colaborating - check if a loader (for carrier agents) or a carrier (for loader agents) is in the same location 
            if agent.has_located and not agent.has_collaborated and not agent.carrying_shelf:
                # if agent is a loader 
                if agent.can_load and not agent.can_carry:
                    # check if a carrier is in the same location
                    if self.grid[_LAYER_AGENTS, agent.y, agent.x]:
                        agent_id_there = self.grid[_LAYER_AGENTS, agent.y, agent.x]
                        agent_there = self.agents[agent_id_there - 1]
                        if agent_there.can_carry and not agent_there.can_load:
                            reward_array[agent.id - 1, 1] = 3/5
                            agent.has_collaborated = True
                            print('collaborated with carrier')
                            # remove reward from agent when it located the shelf (subtask 5)
                            reward_array[agent.id - 1, 4] = -1/5
                            print('removed reward from agent when it located the shelf after colaborating')
                # if agent is a carrier
                elif agent.can_carry and not agent.can_load:
                    # check if a loader is in the same location
                    if self.grid[_LAYER_LOADERS, agent.y, agent.x]:
                        loader_id_there = self.grid[_LAYER_LOADERS, agent.y, agent.x]
                        loader_there = self.agents[loader_id_there - 1]
                        if loader_there.can_load and not loader_there.can_carry:
                            reward_array[agent.id - 1, 1] = 3/7
                            agent.has_collaborated = True
                            print('collaborated with loader')
                            # remove reward from agent when it located the shelf (subtask 5)
                            reward_array[agent.id - 1, 4] = -1/7
                            print('removed reward from agent when it located the shelf after colaborating')

            # subtask 3: loading - load a requested shelf (either independently or colaboratively)              
            # if agent can load independently
            if agent.can_load and agent.can_carry:
                if agent.has_located and not agent.has_loaded and agent.carrying_shelf:
                    shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                    if self.shelfs[shelf_id-1] in self.request_queue:
                        reward_array[agent.id - 1, 2] = 3/5
                        agent.has_loaded = True
                        print('loaded on its own')
                        # remove reward from agent when it located the requested shelf (subtask 5)
                        reward_array[agent.id - 1, 4] = -1/5
                        print('removed reward from agent when it located the shelf after loading')
            # if agent can load collaboratively (carrier agents only)
            if not agent.can_load and agent.can_carry:
                if agent.has_collaborated and not agent.has_loaded and agent.carrying_shelf:
                    shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                    if self.shelfs[shelf_id-1] in self.request_queue:
                        reward_array[agent.id - 1, 2] = 5/7
                        agent.has_loaded = True
                        print('loaded with loader')
                        # remove reward from agent when it collaborated (subtask 5)
                        reward_array[agent.id - 1, 4] = -3/7
                        print('removed reward from agent when it collaborated')
            # if loader agent successfully colaborated in loading the carrier agent
            if agent.can_load and not agent.can_carry:
                if agent.has_collaborated and not agent.has_loaded:
                    # check if a carrier is in the same location
                    if self.grid[_LAYER_AGENTS, agent.y, agent.x]:
                        agent_id_there = self.grid[_LAYER_AGENTS, agent.y, agent.x]
                        agent_there = self.agents[agent_id_there - 1]
                        if agent_there.can_carry and not agent_there.can_load:
                            if agent_there.carrying_shelf in self.request_queue:
                                reward_array[agent.id - 1, 2] = 1
                                agent.has_loaded = True
                                print('helped loading the carrier')
                                # remove reward from agent when it collaborated (subtask 5)
                                reward_array[agent.id - 1, 4] = -3/5
                                print('removed reward from agent when it collaborated')

            # subtask 4: delivery - deliver a shelf to the goal location
            if agent.has_loaded and not agent.has_delivered and agent.carrying_shelf:
                # check if the agent is in the goals location
                if (agent.x, agent.y) in self.goals:
                    reward_array[agent.id - 1, 3] = 1
                    agent.has_delivered = True
                    print('delivered shelf')
                    # remove reward from agent when it loaded the shelf (subtask 5)
                    if agent.can_carry and not agent.can_load:
                        reward_array[agent.id - 1, 4] = -5/7
                    else:
                        reward_array[agent.id - 1, 4] = -3/5
                    print('removed reward from agent when it loaded the shelf after delivering')
                    # replace delivered shelf from environment and refill request queue
                    shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                    agent.carrying_shelf = None
                    self.request_queue.remove(self.shelfs[shelf_id-1])
                    self.removed_shelf_ids.append(shelf_id)
                    self.shelfs[shelf_id-1].id = 0
                    
            self._recalc_grid()

        # rewards are halved for the agents that only can_carry and can_load (to make sure each item delivery receives a reward of 1)
        for i,agent in enumerate(self.agents):
            if agent.can_carry and not agent.can_load:
                reward_array[i] = reward_array[i] * 1/2
            if not agent.can_carry and agent.can_load:
                reward_array[i] = reward_array[i] * 1/2
            
        # reward vector is obtained by point-wise product of reward_array and subtasks_mask, and then summing over the subtasks
        rewards_vector = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            rewards_vector[i] = np.sum(reward_array[i] * self.subtasks_mask[i])

        return rewards_vector, reward_array


    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        # agents that can_carry should not collide
        carry_agents = [ agent for agent in self.agents if agent.can_carry ]
        self.resolve_move_conflict(carry_agents, self.grid, self.agents)

        # transition function
        self.transition_function()

        # reward function (subtasks reward allocation)
        rewards, reward_array = self.reward_function()

        if (self.max_steps and self._cur_steps >= self.max_steps) or (len(self.request_queue)==0):
            dones = self.n_agents * [True]
        else:
            dones = self.n_agents * [False]

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}
        info['reward_subtask_array'] = reward_array

        return new_obs, list(rewards), dones, info

    def render(self, mode="human"):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        ...
    

if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    import time
    from tqdm import tqdm

    time.sleep(2)

    for _ in tqdm(range(1000000)):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
