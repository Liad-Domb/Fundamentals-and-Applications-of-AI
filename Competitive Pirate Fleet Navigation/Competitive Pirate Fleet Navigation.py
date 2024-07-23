import time

from simulator import Simulator as DeepSimulator
from copy import deepcopy
import random
import math
import utils

class Simulator(DeepSimulator):
    def get_state(self):
        return self.state

def neighbours(map, location):
    if (type(location) == str):
        return []
    rows = len(map)
    cols = len(map[0])
    x, y = location[0], location[1]
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for neighbor in tuple(neighbors):
        if (neighbor[0] < 0 or neighbor[0] >= rows or neighbor[1] < 0
                or neighbor[1] >= cols or map[neighbor[0]][neighbor[1]] == 'I'):
            neighbors.remove(neighbor)
    return neighbors

def insert_dict_multiPlication(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    else:
        dictionary[key] *= value
def tile_marine_expectation(marinePaths):
    marineExpectation = {}
    for path in marinePaths:
        length = len(path)
        if (length == 1):
            marineExpectation[path[0]] = 0.25
            continue
        valueEdges = 1 - (1 / (1.5 * length - 1))
        insert_dict_multiPlication(marineExpectation, path[0], valueEdges)
        insert_dict_multiPlication(marineExpectation, path[length - 1], valueEdges)
        if length > 2:
            valueInside = 1 - (1 / (length - 2 / 3))
            for i in range(1, length - 1):
                insert_dict_multiPlication(marineExpectation, path[i], valueInside)

    return marineExpectation

def priority_map(map, location, tileMarineExpectation):
    priorQ = utils.PriorityQueue()
    locations = set()
    priorQ.append((1.25, location))

    prioMap = [[float("inf") for _ in range(len(row))] for row in map]
    prioMap[location[0]][location[1]] = 1

    while len(priorQ) > 0:
        val, cord = priorQ.pop()
        cord_neighbours = neighbours(map, cord)
        locations.add(cord)

        for nCord in cord_neighbours:
            if nCord not in locations:
                locations.add(nCord)
                symbol = map[nCord[0]][nCord[1]]
                if symbol != "B" and symbol != "S":
                    continue
                nVal = val
                if nCord in tileMarineExpectation:
                    marineDiv = tileMarineExpectation[nCord]
                    nVal= nVal / marineDiv
                prioMap[nCord[0]][nCord[1]] = (1 / nVal)
                priorQ.append((nVal * 1.25, nCord))

    return prioMap

def all_tiles_maps(init_state):
    map = init_state["map"]
    marinePaths = []
    marines = init_state["marine_ships"]
    for marine_data in marines.values():
        marinePaths.append(marine_data["path"])
    tileMarineExpectation = tile_marine_expectation(marinePaths)
    allTilesMaps = dict()
    for i in range(len(map)):
        for j in range(len(map[0])):
            allTilesMaps[(i, j)] = priority_map(map, (i, j), tileMarineExpectation)

    return allTilesMaps

def tiles_value(allTilesMaps, base, pirates, enemies, treasures):
    treasure_available = []
    for value in treasures.values():
        location = value["location"]
        reward = value["reward"]
        if location in enemies:
            enemies[location]["reward"] += reward
        elif location in pirates:
            pirates[location]["reward"] += reward
        else:
            treasure_available.append((location, reward))

    locations_helper = dict()
    for ship in pirates.values():
        ship_capacity = ship["capacity"]
        ship_reward = ship["reward"]
        for location in ship["locations"]:
            if location not in locations_helper:
                sum_treasures_val=0
                for treasure_loc, treasure_reward in treasure_available:
                    treasure_matrix = allTilesMaps[treasure_loc]
                    sum_treasures_val += treasure_reward * treasure_matrix[location[0]][location[1]]
                base_diss = allTilesMaps[base][location[0]][location[1]]
                ship_location_val = (ship_capacity * sum_treasures_val
                                     + (2 - ship_capacity) * ship_reward * base_diss)

                enemy_vals = []
                for enemy_val in enemies.values():
                    enemy_loc = enemy_val["location"]
                    enemy_reward = enemy_val["reward"]
                    enemy_diss = allTilesMaps[enemy_loc][location[0]][location[1]]
                    enemy_val = (ship_reward - enemy_reward) * enemy_diss
                    enemy_val = max(enemy_val, 0)
                    ship_location_val += enemy_val
                    enemy_vals.append((enemy_reward, enemy_diss))
                locations_helper[location] = (sum_treasures_val, base_diss, enemy_vals)

            else:
                sum_treasures_val, base_diss, enemy_vals = locations_helper[location]
                ship_location_val = ((2 - ship_capacity) * sum_treasures_val
                                     + ship_capacity * ship_reward * base_diss)
                for enemy_reward, enemy_diss in enemy_vals:
                    enemy_val = (ship_reward - enemy_reward) * enemy_diss
                    enemy_val = max(enemy_val, 0)
                    ship_location_val += enemy_val

            ship["locations"][location] = ship_location_val



def legal_actions_by_ship(ship_actions, pirate_ship, pirate_data, map, base,
                          treasures, enemy_pirates):
    ship_actions.append(("wait", pirate_ship))

    pirate_neighbours = neighbours(map, pirate_data['location'])
    for cord in pirate_neighbours:
        ship_actions.append(("sail", pirate_ship, cord))

    if pirate_data['capacity'] > 0:
        for treasure, treasure_data in treasures.items():
            treasure_neighbours = neighbours(map, treasure_data['location'])
            if pirate_data['location'] in treasure_neighbours:
                ship_actions.append(("collect", pirate_ship, treasure))

    for enemy_pirate, enemy_data in enemy_pirates.items():
        if (enemy_data["capacity"] < 2
                and pirate_data['location'] == enemy_data["location"]):
            ship_actions.append(("plunder", pirate_ship, enemy_pirate))

    if pirate_data['capacity'] < 2 and pirate_data['location'] == base:
        for treasure, treasure_data in treasures.items():
            if treasure_data["location"] == pirate_ship:
                ship_actions.append(("deposit", pirate_ship, treasure))


def custom_action(type ,is_marine, tileValue, pirate_ship, data = None, additional = None):
    if type == "sail":
        if is_marine:
            return ((type, pirate_ship, data), 0.4 + 0.25 * tileValue)
        return ((type, pirate_ship, data), 4 + 2 * tileValue)

    if type == "wait":
        if is_marine:
            return ((type, pirate_ship), 0.1 + 0.25 * tileValue)
        return ((type, pirate_ship), 1 + 2 * tileValue)

    if type == "collect":
        if is_marine:
            return ((type, pirate_ship, data), 0.1 + 0.25 * tileValue)
        return ((type, pirate_ship, data), (12 * additional[0] + 8 * tileValue) * additional[1])

    if type == "plunder":
        if is_marine:
            return ((type, pirate_ship, data), 0.4 + 0.25 * tileValue)
        return ((type, pirate_ship, data), (8 * additional[0] + 8 * tileValue) * additional[1])

    if type == "deposit":
        if is_marine:
            return ((type, pirate_ship, data), 0.6 + 0.5 * tileValue)
        return ((type, pirate_ship, data), (16 * additional[0] + 8 * tileValue) * additional[1])

def ship_actions_and_weights(ship_actions, pirate_ship, pirate_data, map, base,
                             treasures, enemy_pirates, marine_positions, len_pirates):
    location = pirate_data['location']
    capacity = pirate_data['capacity']
    pirate_neighbours = neighbours(map, location)
    for cord in pirate_neighbours:
        tileValue = pirate_data['locations'][cord]
        is_marine = cord in marine_positions
        ship_actions.append(custom_action("sail", is_marine,
                                          tileValue, pirate_ship, cord))

    tileValue = pirate_data['locations'][location]
    is_marine = location in marine_positions
    ship_actions.append(custom_action("wait", is_marine,
                                          tileValue, pirate_ship))
    if capacity > 0:
        for treasure, treasure_data in treasures.items():
            treasure_neighbours = neighbours(map, treasure_data['location'])
            if location in treasure_neighbours:
                ship_actions.append(custom_action("collect", is_marine,
                                          tileValue, pirate_ship, treasure,
                                                  (treasure_data["reward"], len_pirates)))

    for enemy_pirate, enemy_data in enemy_pirates.items():
        if (enemy_data["capacity"] < 2
                and pirate_data['location'] == enemy_data["location"]):
            ship_actions.append(custom_action("plunder", is_marine,
                                          tileValue, pirate_ship, enemy_pirate,
                                              (enemy_data["reward"], len_pirates)))

    if capacity < 2 and location == base:
        for treasure, treasure_data in treasures.items():
            if treasure_data["location"] == pirate_ship:
                ship_actions.append(custom_action("deposit", is_marine,
                                          tileValue, pirate_ship, treasure,
                                                  (treasure_data["reward"], len_pirates)))


def arrange_actions(all_ship_actions):
    all_actions = set()
    keys = [0] * len(all_ship_actions)
    locked = [None] * len(all_ship_actions)

    nextOuter = False
    while (not nextOuter):
        action = []
        for i in range(len(all_ship_actions)):
            ship_action = all_ship_actions[i][keys[i]]
            action_type = ship_action[0]
            if (action_type == "collect" or action_type == "plunder"):
                action_value = ship_action[2]
                if locked[i] != action_value and action_value in locked:
                    break
                locked[i] = action_value
            action.append(ship_action)
        else:
            all_actions.add(tuple(action))

        outerIndex = 0
        nextOuter = True
        while (nextOuter):
            if outerIndex == len(keys):
                break
            innerIndex = keys[outerIndex] + 1
            if (innerIndex == len(all_ship_actions[outerIndex])):
                keys[outerIndex] = 0
                locked[outerIndex] = None
                outerIndex += 1
            else:
                keys[outerIndex] = innerIndex
                locked[outerIndex] = None
                nextOuter = False
    return all_actions


def arrange_actions_weights(all_ship_actions):
    all_actions_weights = set()
    keys = [0] * len(all_ship_actions)
    locked = [None] * len(all_ship_actions)

    nextOuter = False
    while (not nextOuter):
        action = []
        weight = 0
        for i in range(len(all_ship_actions)):
            ship_combination = all_ship_actions[i][keys[i]]
            ship_action = ship_combination[0]
            ship_weight = ship_combination[1]
            action_type = ship_action[0]
            if (action_type == "collect" or action_type == "plunder"):
                action_value = ship_action[2]
                if locked[i] != action_value and action_value in locked:
                    break
                locked[i] = action_value
            action.append(ship_action)
            weight += ship_weight
        else:
            all_actions_weights.add((tuple(action), weight))

        outerIndex = 0
        nextOuter = True
        while (nextOuter):
            if outerIndex == len(keys):
                break
            innerIndex = keys[outerIndex] + 1
            if (innerIndex == len(all_ship_actions[outerIndex])):
                keys[outerIndex] = 0
                locked[outerIndex] = None
                outerIndex += 1
            else:
                keys[outerIndex] = innerIndex
                locked[outerIndex] = None
                nextOuter = False
    return all_actions_weights

def wait_action_generate(state, player_number):
    action = []
    pirate_ships = state["pirate_ships"]
    for pirate_ship, pirate_data in pirate_ships.items():
        if pirate_data['player'] == player_number:
            action.append(("wait", pirate_ship))
    return tuple(action)

def legal_actions_by_state(state, player_number):
    map = state["map"]
    base = state["base"]
    pirate_ships = state["pirate_ships"]
    treasures = state["treasures"]

    my_pirates = dict()
    enemy_pirates = dict()
    for pirate_ship, pirate_data in pirate_ships.items():
        if pirate_data['player'] == player_number:
            my_pirates[pirate_ship] = {'location': pirate_data['location'],
                                       'capacity': pirate_data['capacity']}
        else:
            enemy_pirates[pirate_ship] = {'location': pirate_data['location'],
                                          'capacity': pirate_data['capacity'], }
    all_ship_actions = []
    for pirate_ship, pirate_data in my_pirates.items():
        ship_actions = []
        legal_actions_by_ship(ship_actions, pirate_ship, pirate_data, map, base,
                              treasures, enemy_pirates)
        all_ship_actions.append(ship_actions)

    return arrange_actions(all_ship_actions)


def actions_with_weights(state, player_number, allTilesMaps):
    map = state["map"]
    base = state["base"]
    pirate_ships = state["pirate_ships"]
    treasures = state["treasures"]
    marines = state["marine_ships"]

    my_pirates = dict()
    enemy_pirates = dict()
    for pirate_ship, pirate_data in pirate_ships.items():
        if pirate_data['player'] == player_number:
            pirate_location = pirate_data['location']
            pirate_neighbours = neighbours(map, pirate_location)
            pirate_neigh_locs = {loc : None for loc in pirate_neighbours}
            pirate_neigh_locs[pirate_location] = 0
            my_pirates[pirate_ship] = {'location': pirate_location,
                                       'capacity': pirate_data['capacity'],
                                       'reward': 0,
                                       'locations': pirate_neigh_locs}
        else:
            enemy_pirates[pirate_ship] = {'location': pirate_data['location'],
                                          'capacity': pirate_data['capacity'],
                                          'reward':0}

    marine_positions = set()
    for marine in marines.values():
        marine_positions.add(marine["path"][marine["index"]])

    tiles_value(allTilesMaps, base, my_pirates, enemy_pirates, treasures)
    all_ship_actions = []
    len_pirates = len(pirate_neighbours)
    for pirate_ship, pirate_data in my_pirates.items():
        ship_actions = []
        ship_actions_and_weights(ship_actions, pirate_ship, pirate_data, map, base,
                                 treasures, enemy_pirates, marine_positions, len_pirates)
        all_ship_actions.append(ship_actions)

    return arrange_actions_weights(all_ship_actions)

def rand_action_by_weight(state, player_number, allTilesMaps):
    all_actions_weights = actions_with_weights(state, player_number, allTilesMaps)
    actions, weights = zip(*all_actions_weights)
    return random.choices(actions, weights=weights, k=1)[0]

def conclude_enemy_action(before_state, after_state, player_number, plundered):
    base = before_state["base"]
    treasures_b = before_state["treasures"]
    treasures_a = after_state["treasures"]
    pirates_b = before_state["pirate_ships"]
    pirates_a = after_state["pirate_ships"]
    ships_p = []
    ships_e = []
    action = []
    lst_plunder = []
    for pirate_ship, pirate_data in pirates_a.items():
        if pirate_data['player'] == player_number:
            ships_p.append(pirate_ship)
        else:
            ships_e.append(pirate_ship)

    for ship in ships_e:
        not_found = True
        location = pirates_a[ship]["location"]
        if location != pirates_b[ship]["location"]:
            action.append(("sail", ship, location))
            not_found = False
        if not_found and location == base and ship not in plundered:
            for treasure_b, tr_data_b in treasures_b.items():
                if tr_data_b['location'] == ship:
                    if treasure_b not in treasures_a:
                        action.append(("deposit", ship, treasure_b))
                        not_found = False
                        break
        if not_found:
            for treasure_a, tr_data_a in treasures_a.items():
                if tr_data_a['location'] == ship:
                    if treasure_a not in treasures_b:
                        action.append(("collect", ship, treasure_a))
                        not_found = False
                        break
        if not_found:
            for p_ship in ships_p:
                if pirates_a[p_ship]['location'] == location and p_ship not in lst_plunder:
                    for treasure_b, tr_data_b in treasures_b.items():
                        if tr_data_b['location'] == p_ship:
                            if treasure_b not in treasures_a:
                                action.append(("plunder", ship, p_ship))
                                lst_plunder.append(p_ship)
                                not_found = False
                                break
        if not_found:
            action.append(("wait", ship))
    return tuple(action)


class UCTNode:
    """
    A class for a single node. not mandatory to use but may help you.
    """

    def __init__(self, player_number, action=None, parent=None):
        self.player_number = player_number
        self.action = action
        self.parent = parent
        self.value = 0
        self.times = 0
        self.children_actions = set()
        self.children = []

    def get_times(self):
        return self.times

    def get_value(self):
        return self.value

    def get_action(self):
        return self.action

    def get_player_number(self):
        return self.player_number

    def get_children(self):
        return self.children

    def remove_parent(self):
        self.parent = None

    def reset_params(self):
        self.value = 0
        self.times = 0

    def is_new_eval_better(self, current_eval, new_eval):
        if self.player_number == 1:
            return new_eval > current_eval
        else:
            return current_eval > new_eval

    def calc_eval(self, parent_time, node_time, node_value):
        if self.player_number == 1:
            return ((node_value / node_time) +
                    math.sqrt(2 * math.log(parent_time) / node_time))
        return ((node_value / node_time) -
                math.sqrt(2 * math.log(parent_time) / node_time))

    def reassure_selection(self, state):
        p_num = self.player_number
        p_opp = ((p_num - 2) % 2) + 1  # Exchanges 1 -> 2 -> 1

        all_actions = legal_actions_by_state(state, p_num)
        expander_actions = all_actions - self.children_actions
        forbidden_actions = self.children_actions - all_actions
        expander_children = []
        for action in expander_actions:
            self.children_actions.add(action)
            child = UCTNode(p_opp, action, self)
            self.children.append(child)
            expander_children.append(child)
        return expander_children, forbidden_actions

    def has_children(self):
        return len(self.children_actions) > 0

    def remove_forbidden(self, forbidden_actions):
        self.children_actions = self.children_actions - forbidden_actions
        self.children = [child for child in self.children if child.get_action() not in forbidden_actions]

    def select_child(self, forbidden_actions=None):
        if forbidden_actions is None:
            forbidden_actions = set()
        selected_child = self.children[0]
        i = 1
        while selected_child.get_action() in forbidden_actions:
            selected_child = self.children[i]
            i += 1

        current_eval = self.calc_eval(self.times, selected_child.get_times(),
                                      selected_child.get_value())
        for child in self.children[i:]:
            if child.get_action() not in forbidden_actions:
                new_eval = self.calc_eval(self.times, child.get_times(),
                                          child.get_value())
                if self.is_new_eval_better(current_eval, new_eval):
                    selected_child = child
                    current_eval = new_eval
        return selected_child

    def expand_children(self, state):
        p_num = self.player_number
        p_opp = ((p_num - 2) % 2) + 1  # Exchanges 1 -> 2 -> 1

        self.children_actions = legal_actions_by_state(state, p_num)
        for action in self.children_actions:
            self.children.append(UCTNode(p_opp, action, self))
        return self.children

    def insert_and_back(self, score):
        self.times += 1
        self.value += score["player 1"] - score["player 2"]
        if self.parent is not None:
            self.parent.insert_and_back(score)


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self, player_number, init_state, turns_to_go):
        self.player_number = player_number
        self.init_state = init_state
        self.turns_to_go = turns_to_go
        self.root = UCTNode(self.player_number)
        root_children = self.expansion(self.root, self.init_state)
        simulator = Simulator(self.init_state)
        self.simulation_backpropagation(simulator, root_children, self.turns_to_go, self.player_number)

    def simulate(self, simulator, first_action, turns_to_go, player_number):
        sim = deepcopy(simulator)
        if player_number == 1:
            sim.act(first_action, 1)
            state = sim.get_state()
            all_actions = legal_actions_by_state(state, 2)
            action = random.choice(list(all_actions))
            sim.act(action, 2)
            sim.check_collision_with_marines()
            sim.move_marines()
        else:
            sim.act(first_action, 2)
            sim.check_collision_with_marines()
            sim.move_marines()
        turns_to_go -= 1
        for i in range(turns_to_go):
            for j in range(1, 3):
                state = sim.get_state()
                all_actions = legal_actions_by_state(state, j)
                action = random.choice(list(all_actions))
                sim.act(action, j)

            sim.check_collision_with_marines()
            sim.move_marines()

        return sim.get_score()

    def simulation_backpropagation(self, simulator, action_nodes, turns_to_go, player_number):
        for node in action_nodes:
            node.insert_and_back(self.simulate(simulator, node.get_action(),
                                               turns_to_go, player_number))

    def expansion(self, node, state):
        return node.expand_children(state)

    def selection(self, stop=False):
        simulator = Simulator(self.init_state)
        turns = self.turns_to_go

        if turns == 1:
            expander_children, forbidden_actions = self.root.reassure_selection(self.init_state)
            self.root.remove_forbidden(forbidden_actions)
            self.simulation_backpropagation(Simulator(self.init_state), expander_children,
                                            self.turns_to_go, self.player_number)
            selected_node = self.root.select_child(forbidden_actions)
            return selected_node, simulator, turns

        selected_node = self.root.select_child()
        if stop:
            return selected_node, simulator, turns

        p_num = self.root.get_player_number()
        while turns > 0:
            action = selected_node.get_action()
            simulator.act(action, p_num)
            if p_num == 2:
                simulator.check_collision_with_marines()
                simulator.move_marines()
                turns -= 1
            if selected_node.has_children():
                p_num = selected_node.get_player_number()
                state = simulator.get_state()
                expander_children, forbidden_actions = selected_node.reassure_selection(state)
                self.simulation_backpropagation(simulator, expander_children, turns, p_num)
                selected_node = selected_node.select_child(forbidden_actions)
            else:
                break

        return selected_node, simulator, turns

    def round(self):
        node, simulator, turns = self.selection()
        state = simulator.get_state()
        children = self.expansion(node, state)
        self.simulation_backpropagation(simulator, children, turns, node.get_player_number())

    def action(self):
        node, simulator, turns = self.selection(True)
        return node.get_action()


class AgentTree(UCTTree):

    def __init__(self, player_number, init_state, turns_to_go, allTilesMaps, root_action = None):
        self.player_number = player_number
        self.init_state = init_state
        self.turns_to_go = turns_to_go
        self.allTilesMaps = allTilesMaps
        self.root = UCTNode(self.player_number, root_action, None)
        root_children = self.expansion(self.root, self.init_state)
        simulator = Simulator(self.init_state)
        self.simulation_backpropagation(simulator, root_children, self.turns_to_go, self.player_number)

    def get_root(self):
        return self.root

    def action(self):
        node, simulator, turns = self.selection(True)
        return node

    def advance_root(self, chosen_node, before_state, after_state, plundered):
        self.turns_to_go -= 1
        self.init_state = after_state
        enemy_action = conclude_enemy_action(before_state, after_state, self.player_number, plundered)
        next_root = None
        for child in chosen_node.get_children():
            if child.get_action() == enemy_action:
                next_root = child
                next_root.remove_parent()
                break
        if next_root is None:
            next_root = UCTNode(self.player_number)
        self.root = next_root
        if not self.root.has_children():
            self.root.reset_params()
            root_children = self.expansion(self.root, after_state)
            simulator = Simulator(after_state)
            self.simulation_backpropagation(simulator, root_children, self.turns_to_go, self.player_number)
        else:
            expander_children, forbidden_actions = self.root.reassure_selection(self.init_state)
            self.root.remove_forbidden(forbidden_actions)
            self.simulation_backpropagation(Simulator(self.init_state), expander_children,
                                            self.turns_to_go, self.player_number)

    def calculate_turns(self, turns_to_go):  # liad_change
        if 50 <= turns_to_go <= 100:
            # Take a lower value of 2 thirds of turns_to_go
            return (turns_to_go * 2) // 3
        elif 10 < turns_to_go < 50:
            # Take a lower value of half of turns_to_go
            return (turns_to_go * 3) // 4
        else:
            # If under 10 (including 10), take all of turns_to_go
            return turns_to_go

    def simulate(self, simulator, first_action, turns_to_go, player_number):

        # liad_change:
        rule_turns = self.calculate_turns(turns_to_go)

        sim = deepcopy(simulator)
        if player_number == 1:
            sim.act(first_action, 1)
            state = sim.get_state()
            sim.act(rand_action_by_weight(state, 2, self.allTilesMaps), 2)
            sim.check_collision_with_marines()
            sim.move_marines()
        else:
            sim.act(first_action, 2)
            sim.check_collision_with_marines()
            sim.move_marines()
        rule_turns -= 1

        for i in range(rule_turns):
            for j in range(1, 3):
                state = sim.get_state()
                sim.act(rand_action_by_weight(state, j, self.allTilesMaps), j)

            sim.check_collision_with_marines()
            sim.move_marines()

        return sim.get_score()

class Agent:
    def __init__(self, initial_state, player_number):
        self.start_time = time.perf_counter()
        self.ids = IDS
        self.player_number = player_number
        self.turns_to_go = initial_state["turns to go"]//2
        self.allTilesMaps = all_tiles_maps(initial_state)
        self.before_state = initial_state
        if self.player_number == 1:
            self.tree = AgentTree(self.player_number, initial_state, self.turns_to_go, self.allTilesMaps)
            self.chosen_node = None
        else:
            self.tree = AgentTree(self.player_number, initial_state, self.turns_to_go, self.allTilesMaps,
                                  wait_action_generate(initial_state, 2))
            self.chosen_node = self.tree.get_root()

        self.tree_round(59.89)  # change

        self.plundered = []

    def handle_time(self):
        time_measure = time.perf_counter()
        round_time = time_measure - self.start_time
        self.start_time = time_measure
        self.time_left -= round_time
        if round_time * 1.21 > self.round_time:
            self.round_time = round_time * 1.21

    def tree_round(self, time_left):
        self.time_left = time_left
        self.round_time = 0
        self.handle_time()
        while self.time_left > self.round_time:
            self.tree.round()
            self.handle_time()

    def act(self, state):
        self.start_time = time.perf_counter()
        if self.chosen_node is not None:
            self.tree.advance_root(self.chosen_node, self.before_state, state, self.plundered)
        self.plundered = []
        self.tree_round(4.89)
        self.turns_to_go -= 1
        self.before_state = state
        self.chosen_node = self.tree.action()
        action = self.chosen_node.get_action()
        for atomic in action:
            if atomic[0] == "plunder":
                self.plundered.append(atomic[2])
        return self.chosen_node.get_action()


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.turns_to_go = initial_state["turns to go"]//2

    def handle_time(self):
        time_measure = time.perf_counter()
        round_time = time_measure - self.start_time
        self.start_time = time_measure
        self.time_left -= round_time
        if round_time * 1.21 > self.round_time:
            self.round_time = round_time * 1.21

    def act(self, state):
        self.start_time = time.perf_counter()
        self.time_left = 4.89
        self.round_time = 0
        self.tree = UCTTree(self.player_number, state, self.turns_to_go)
        self.handle_time()
        while self.time_left > self.round_time:
            self.tree.round()
            self.handle_time()
        self.turns_to_go -= 1
        return self.tree.action()
