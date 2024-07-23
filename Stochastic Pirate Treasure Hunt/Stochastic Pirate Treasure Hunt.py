import itertools
import copy
import json
import networkx as nx
import random
import time
import utils

#___________________________________OPTIMAL__________________________________
class OptimalPirateAgent:

    def state_to_dict(self, data_dict):
        new_data = {}
        keys_to_get = ["pirate_ships", "treasures", "marine_ships"]
        for key in keys_to_get:
            if key in data_dict:
                new_data[key] = data_dict[key]
        return new_data

    def create_all_states(self, original_state):
        our_map = original_state["map"]
        treasures = original_state["treasures"]
        pirates = original_state["pirate_ships"]
        marine_ships = original_state["marine_ships"]

        ships_can_be = []
        for row in range(len(our_map)):
            for col in range(len(our_map[0])):
                if our_map[row][col] != 'I':
                    ships_can_be.append((row, col))

        pirate_positions_comb = list(itertools.product(ships_can_be, repeat=len(pirates)))
        capacity_comb = []

        for pirate in pirates.values():
            capacity_comb.append(range(pirate["capacity"] + 1))

        possible_states = []
        treasure_positions_comb = itertools.product(
            *[treasure["possible_locations"] for treasure in treasures.values()])
        marine_indexes_combinations = itertools.product(
            *[range(len(navy_ship["path"])) for navy_ship in marine_ships.values()])

        for pirate_combination, capacity_combination, treasure_combination, marine_index_combination \
                in itertools.product(pirate_positions_comb, itertools.product(*capacity_comb), treasure_positions_comb,
                                     marine_indexes_combinations):
            new_state = copy.deepcopy(original_state)

            for (location, capacity), pirate_ship_name in zip(zip(pirate_combination, capacity_combination),
                                                              pirates.keys()):
                new_state["pirate_ships"][pirate_ship_name]["capacity"] = capacity
                new_state["pirate_ships"][pirate_ship_name]["location"] = location

            for idx, marine_ship_name in zip(marine_index_combination, marine_ships.keys()):
                new_state["marine_ships"][marine_ship_name]["index"] = idx

            for location, treasure_name in zip(treasure_combination, treasures.keys()):
                if (location in self.initial["treasures"][treasure_name]["possible_locations"]):
                    new_state["treasures"][treasure_name]["location"] = location

            possible_states.append(self.state_to_dict(new_state))

        return possible_states

    def create_all_combinations(self, current_state):
        combined_locations_and_probabilities = []
        for treasure_key, treasure_dict in current_state['treasures'].items():
            treasure_location_probability_pairs = []

            for location in treasure_dict['possible_locations']:
                treasure_probability = treasure_dict['prob_change_location']
                if location != treasure_dict['location']:
                    probability = treasure_probability / len(treasure_dict['possible_locations'])
                else:
                    probability = treasure_probability / len(treasure_dict['possible_locations']) + (
                                1 - treasure_probability)

                treasure_location_probability_pairs.append((treasure_key, location, probability))
            combined_locations_and_probabilities.append(treasure_location_probability_pairs)

        for marine_name, marine_dict in current_state['marine_ships'].items():
            marine_location_probability_pairs = []
            curr_index = marine_dict['index']
            if len(marine_dict['path']) == 1:
                prob = 1
                marine_location_probability_pairs += [(marine_name, curr_index, prob)]
            elif (curr_index == 0):
                prob = 1 / 2
                marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                marine_location_probability_pairs += [(marine_name, curr_index + 1, prob)]
            elif (curr_index == len(marine_dict['path']) - 1):
                prob = 1 / 2
                marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                marine_location_probability_pairs += [(marine_name, curr_index - 1, prob)]
            else:
                prob = 1 / 3
                marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                marine_location_probability_pairs += [(marine_name, curr_index - 1, prob)]
                marine_location_probability_pairs += [(marine_name, curr_index + 1, prob)]

            combined_locations_and_probabilities.append(marine_location_probability_pairs)

        combinations_all = list(itertools.product(*combined_locations_and_probabilities))

        final_combinations_all = []
        for comb in combinations_all:
            probabilities = [item[-1] for item in comb]
            product_of_probabilities = 1.0
            for prob in probabilities:
                product_of_probabilities *= prob
            result = (product_of_probabilities,) + comb
            final_combinations_all.append(result)
        return final_combinations_all

    def all_actions_for_state(self, state):
        all_ship_actions = []
        pirate_ships = state["pirate_ships"]
        all_treasures = state["treasures"]

        for pirate_ship_name, pirate_ship_dict in pirate_ships.items():
            actions_per_ship=[]

            cord_x = pirate_ship_dict["location"][0]
            cord_y = pirate_ship_dict["location"][1]
            possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1),
                              (cord_x, cord_y - 1)]

            # sail
            for move in possible_moves:
                if self.is_valid_sail(move[0], move[1]):
                    actions_per_ship+=[("sail", pirate_ship_name, move)]

            # collect
            if pirate_ship_dict["capacity"] > 0:
                near_islands_list = []
                for move in possible_moves:
                    if self.is_adj_island(move[0], move[1]):
                        near_islands_list.append((move[0], move[1]))


                for treasure_name, treasure_dict in all_treasures.items():
                    if treasure_dict["location"] in near_islands_list:
                        actions_per_ship+=[("collect", pirate_ship_name, treasure_name)]

            # deposit
            if (self.is_valid_deposit(cord_x, cord_y) and pirate_ship_dict['capacity'] <self.initial['pirate_ships'][pirate_ship_name]['capacity']):
                actions_per_ship+=[("deposit", pirate_ship_name)]

            # wait
            actions_per_ship+=[("wait", pirate_ship_name)]

            all_ship_actions.append(actions_per_ship)

        all_combinations = list(itertools.product(*all_ship_actions))
        all_combinations += ["reset"]
        return all_combinations

    def is_valid_sail(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] in ['S', 'B']):
            return True
        return False

    def is_valid_deposit(self, x, y):
        if (self.map[x][y] == 'B'):
            return True
        return False

    def is_adj_island(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] == 'I'):
            return True
        return False

    def next_state_prob_and_points(self, current_state, action):
        new_state_initial = copy.deepcopy(current_state)

        #reset
        if action == "reset":
            new_state = copy.deepcopy(self.initial)
            point_to_change = -2

            marine_ship_locations_list = []
            for marine in new_state['marine_ships'].values():
                marine_ship_locations_list.append(marine['path'][marine['index']])

            for ship_name, ship in new_state['pirate_ships'].items():
                if ship['location'] in marine_ship_locations_list:
                    point_to_change -= 1
                    new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name]['capacity']

            return [(self.state_to_dict(new_state), 1, point_to_change)]

        #all other actions:
        else:
            for current_action in action:
                if current_action[0] == 'sail':
                    new_state_initial['pirate_ships'][current_action[1]]['location'] = current_action[2]

                if current_action[0] == 'collect':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] -= 1

                if current_action[0] == 'deposit':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] = self.initial['pirate_ships'][current_action[1]]['capacity']
            combinations = self.create_all_combinations(current_state)

            next_state_prob_and_points_list = []
            for comb in combinations:
                new_state = copy.deepcopy(new_state_initial)
                for item in comb[1:]:
                    if item[0] in self.treasures_names:
                        new_state['treasures'][item[0]]['location'] = item[1]
                    else:
                        if item[0] in self.marines_names:
                            new_state['marine_ships'][item[0]]['index'] = item[1]

                marine_distances_and_locations = []
                for marine_item in new_state['marine_ships'].values():
                    marine_distances_and_locations.append(marine_item['path'][marine_item['index']])

                point_to_change = 0
                for current_action in action:
                    if current_action[0] == "deposit":
                        point_to_change += 4 * (self.initial['pirate_ships'][current_action[1]]['capacity'] - current_state['pirate_ships'][current_action[1]]['capacity'])

                for pirate_name, pirate_dict in new_state['pirate_ships'].items():
                    if pirate_dict['location'] in marine_distances_and_locations:
                        point_to_change -= 1
                        new_state['pirate_ships'][pirate_name]['capacity'] = self.initial['pirate_ships'][pirate_name]['capacity']

                next_state_prob_and_points_list += [(self.state_to_dict(new_state), comb[0], point_to_change)]

            return next_state_prob_and_points_list

    def VI(self):
        value_iteration_matrix = {}
        all_the_possible_states = self.create_all_states(self.initial)

        for i in range(self.turns_to_go + 1):
            value_iteration_matrix[i] = {json.dumps(state): 0 for state in all_the_possible_states}

        actions_from_state_dict = {}
        next_state_dict = {}
        policy = {}

        for state in all_the_possible_states:
            actions_from_state_dict[json.dumps(state)] = self.all_actions_for_state(state)

        for state in all_the_possible_states:
            state_to_string = json.dumps(state)
            actions_list = self.all_actions_for_state(state)
            action_dictionary = {}
            for action in actions_list:
                action_dictionary[action] = self.next_state_prob_and_points(state, action)

            next_state_dict[state_to_string] = action_dictionary

        for state in all_the_possible_states:
            policy[json.dumps(state)] = dict.fromkeys(range(1, self.turns_to_go + 1), "not_filled_yet")
        for turn_to_go_idx in range(1, self.turns_to_go + 1):
            for state in all_the_possible_states:
                maximum_val = float('-inf')
                best_policy = None
                actions_after_state_list = actions_from_state_dict[json.dumps(state)]
                for action in actions_after_state_list:
                    value_after_state_and_action = 0
                    for next_state, probability, points_to_change in next_state_dict[json.dumps(state)][action]:
                        value_after_state_and_action += probability * ((value_iteration_matrix[turn_to_go_idx - 1][json.dumps(next_state)]) + points_to_change)
                    if value_after_state_and_action > maximum_val:
                        maximum_val = value_after_state_and_action
                        best_policy = action
                if (maximum_val < 0):
                    maximum_val = 0
                    best_policy = "terminate"
                value_iteration_matrix[turn_to_go_idx][json.dumps(state)] = maximum_val
                policy[json.dumps(state)][turn_to_go_idx] = best_policy

        return policy

    def __init__(self, initial):
        self.initial = initial
        self.turns_to_go = initial["turns to go"]
        self.marines_names = list(initial["marine_ships"].keys())
        self.treasures_names = list(initial["treasures"].keys())
        self.base_location = next(iter(initial['pirate_ships'].values()), None).get("location")
        self.map = initial['map']
        self.grid_rows = len(self.map)
        self.grid_cols = len(self.map[0])
        self.round_num = 0
        self.policy = self.VI()

    def act(self, state):
        self.round_num = self.round_num + 1
        return self.policy[json.dumps(self.state_to_dict(state))][self.turns_to_go - self.round_num + 1]

#_____________________________________________AGENT____________________________________________________
class PirateAgent:

    def state_to_dict(self, data_dict):
        keys_to_get = ["pirate_ships", "treasures", "marine_ships"]
        change_key = "pirate_ships"
        new_data = {}
        for key in keys_to_get:
            new_data[key] = data_dict[key]
        new_data[change_key] = {self.first_pirate_name: data_dict[change_key][self.first_pirate_name]}
        return new_data

    def actions_for_state(self, state):
        all_ship_actions = []
        pirate_ships = state["pirate_ships"]
        all_treasures = state["treasures"]

        for pirate_ship_name, pirate_ship_dict in pirate_ships.items():
            actions_per_ship = []

            cord_x = pirate_ship_dict["location"][0]
            cord_y = pirate_ship_dict["location"][1]
            possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1),
                              (cord_x, cord_y - 1)]

            # sail
            for move in possible_moves:
                if self.is_valid_sail(move[0], move[1]):
                    actions_per_ship += [("sail", pirate_ship_name, move)]

            # collect
            if pirate_ship_dict["capacity"] > 0:
                near_islands_list = []
                for move in possible_moves:
                    if self.is_adj_island(move[0], move[1]):
                        near_islands_list.append((move[0], move[1]))

                for treasure_name, treasure_dict in all_treasures.items():
                    if treasure_dict["location"] in near_islands_list:
                        actions_per_ship += [("collect", pirate_ship_name, treasure_name)]

            # deposit
            if (self.is_valid_deposit(cord_x, cord_y) and pirate_ship_dict['capacity'] <
                    self.initial['pirate_ships'][pirate_ship_name]['capacity']):
                actions_per_ship += [("deposit", pirate_ship_name)]

            # wait
            actions_per_ship += [("wait", pirate_ship_name)]

            all_ship_actions.append(actions_per_ship)

        all_combinations = list(itertools.product(*all_ship_actions))
        all_combinations += ["reset"]
        return all_combinations

    def is_valid_sail(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] in ['S', 'B']):
            return True
        return False

    def is_valid_deposit(self, x, y):
        if (self.map[x][y] == 'B'):
            return True
        return False

    def is_adj_island(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] == 'I'):
            return True
        return False

    def environment_step(self, state):
        for t in state['treasures']:
            treasure_stats = state['treasures'][t]
            if random.random() < treasure_stats['prob_change_location']:
                treasure_stats['location'] = random.choice(
                    treasure_stats['possible_locations'])

        for marine in state['marine_ships']:
            marine_stats = state["marine_ships"][marine]
            index = marine_stats["index"]
            if len(marine_stats["path"]) == 1:
                continue
            if index == 0:
                marine_stats["index"] = random.choice([0, 1])
            elif index == len(marine_stats["path"])-1:
                marine_stats["index"] = random.choice([index, index-1])
            else:
                marine_stats["index"] = random.choice(
                    [index-1, index, index+1])
        return

    def next_state_and_points(self, current_state, action):
        new_state_initial = copy.deepcopy(current_state)
        self.environment_step(new_state_initial)

        if action == "reset":
            new_state = copy.deepcopy(self.initial)
            point_to_change = -2

            marine_ship_locations_list = []
            for marine in new_state['marine_ships'].values():
                marine_ship_locations_list.append(marine['path'][marine['index']])

            for ship_name, ship in new_state['pirate_ships'].items():

                if ship['location'] in marine_ship_locations_list:
                    point_to_change -= 1
                    new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name][
                        'capacity']

            return [(self.state_to_dict(new_state), point_to_change)]

        else:
            point_to_change = 0
            marine_ship_locations_list = []
            for marine in new_state_initial['marine_ships'].values():
                marine_ship_locations_list.append(marine['path'][marine['index']])

            for current_action in action:
                if current_action[0] == 'sail':
                    new_state_initial['pirate_ships'][current_action[1]]['location'] = current_action[2]

                if current_action[0] == 'collect':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] -= 1

                if current_action[0] == 'deposit':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] = \
                    self.initial['pirate_ships'][current_action[1]]['capacity']

            for current_action in action:
                if current_action[0] == "deposit":
                    point_to_change += self.num_of_pirates * 4 * (self.initial['pirate_ships'][current_action[1]]['capacity'] -
                                            current_state['pirate_ships'][current_action[1]]['capacity'])

            for pirate_name, pirate_dict in new_state_initial['pirate_ships'].items():
                if pirate_dict['location'] in marine_ship_locations_list:
                    point_to_change -= self.num_of_pirates * 1

                    new_state_initial['pirate_ships'][pirate_name]['capacity'] = \
                    self.initial['pirate_ships'][pirate_name]['capacity']

            return [(self.state_to_dict(new_state_initial), point_to_change)]

    def update_states(self, dict_of_states, state_key, action, points, dist):

        if state_key not in dict_of_states:
            dict_of_states[state_key] = {}

        if action not in dict_of_states[state_key]:
            dict_of_states[state_key][action] = {'points': points / dist, 'distance': dist, 'count': 1}

        else:
            dict_of_states[state_key][action]['points'] += points / dist
            dict_of_states[state_key][action]['distance'] += dist
            dict_of_states[state_key][action]['count'] += 1

    def deposite_dict(self, points_from_action, states_dict):
        length= len(points_from_action)

        for dist in range(1, length):
            state, action, points = points_from_action[length - dist]
            key = json.dumps(state, sort_keys=True)
            points_from_action[length - dist - 1][2] += points
            self.update_states(states_dict, key, action, points, dist)

        state, action, points = points_from_action[0]
        key = json.dumps(state, sort_keys=True)
        self.update_states(states_dict, key, action, points, dist=length)
        points_from_action.clear()

    def first_pick_policy(self, curr_state, policy, probability, first_prob):
        selected_action = None

        if random.random() < probability:
            state_key = json.dumps(curr_state, sort_keys=True)
            if state_key in policy:
                selected_action = policy[state_key]["best_action"]

        elif random.random() < first_prob:
            capacity = curr_state["pirate_ships"][self.ships_names[0]]["capacity"]
            list_of_actions = self.actions_for_state(curr_state)
            if capacity == 1:
                selected_action = self.our_h(list_of_actions, True, curr_state["treasures"])
            elif capacity == 2:
                selected_action = self.our_h(list_of_actions, False, curr_state["treasures"])
            else:
                selected_action = self.our_h(list_of_actions, True, False)

        return selected_action

    def not_first_pick_policy(self, curr_state, policy, probability, first_prob):
        selected_action = None
        if random.random() < first_prob:
            capacity = curr_state["pirate_ships"][self.ships_names[0]]["capacity"]
            list_of_actions = self.actions_for_state(curr_state)
            if capacity == 1:
                selected_action = self.our_h(list_of_actions, True, curr_state["treasures"])
            elif capacity == 2:
                selected_action = self.our_h(list_of_actions, False, curr_state["treasures"])
            else:
                selected_action = self.our_h(list_of_actions, True, False)

        elif random.random() < probability:
            state_key = json.dumps(curr_state, sort_keys=True)
            if state_key in policy:
                selected_action = policy[state_key]["best_action"]
        return selected_action

    def one_turn_policy(self, curr_state, points_from_action, policy, probability, first_prob, first_policy=True):
        if (first_policy==False):
            selected_action = self.not_first_pick_policy(curr_state, policy,probability, first_prob)
        else:
            selected_action = self.first_pick_policy(curr_state, policy,probability, first_prob)

        if selected_action == None:
            list_of_actions = self.actions_for_state(curr_state)
            selected_action = random.choice(list_of_actions)
        result = self.next_state_and_points(curr_state, selected_action)
        next_state, points = result[0]
        points_from_action.append([curr_state, selected_action, points])
        return next_state, selected_action

    def one_turn_no_policy(self, curr_state, points_from_action, first_prob):
        selected_action = None
        list_of_actions = self.actions_for_state(curr_state)

        if random.random() < first_prob:
            capacity = curr_state["pirate_ships"][self.ships_names[0]]["capacity"]
            if capacity == 1:
                selected_action = self.our_h(list_of_actions, True, curr_state["treasures"])
            elif capacity == 2:
                selected_action = self.our_h(list_of_actions, False, curr_state["treasures"])
            else:
                selected_action = self.our_h(list_of_actions, True, False)
        else:
            for action in list_of_actions:
                if action[0][0] == "deposit" and selected_action is None:
                    selected_action = action
                elif action[0][0] == "collect":
                    if random.random() > 0.05:
                        selected_action = action

        if selected_action == None:
            selected_action = random.choice(list_of_actions)
        result = self.next_state_and_points(curr_state, selected_action)
        next_state, points = result[0]
        points_from_action.append([curr_state, selected_action, points])
        return next_state, selected_action

    def deposit_sim_policy(self, policy, probability, first_prob=0.3, last_prob=0.9):
        curr_state = self.state_to_dict(self.initial)
        states_dict = {}
        points_from_action = []
        turns_to_go_change = [0,max(self.turns_to_go // 2, self.turns_to_go - 3 * self.base_long_distance, 1)]
        turns_to_go_change.append(max(self.turns_to_go - (self.base_long_distance // 2), self.turns_to_go - 10, turns_to_go_change[1] + 1))
        turns_to_go_change.append(max(self.turns_to_go, turns_to_go_change[2] + 1))
        turns_to_go_change.append(max(self.turns_to_go + self.base_long_distance, turns_to_go_change[3] + 1))

        for i in range(turns_to_go_change[0], turns_to_go_change[1]):
            next_state, selected_action = self.one_turn_policy(curr_state,points_from_action,policy,probability,first_prob)

            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
            curr_state = next_state

        first_prob_change = (last_prob - first_prob) / (turns_to_go_change[2] - turns_to_go_change[1])
        first_prob += first_prob_change
        for i in range(turns_to_go_change[1], turns_to_go_change[2]):
            next_state, selected_action = self.one_turn_policy(curr_state,points_from_action,policy,probability,first_prob)
            first_prob += first_prob_change
            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
            curr_state = next_state

        first_prob_change = (last_prob - first_prob) / (turns_to_go_change[3] - turns_to_go_change[2])
        first_prob += first_prob_change
        for i in range(turns_to_go_change[2], turns_to_go_change[3]):
            next_state, selected_action = self.one_turn_policy(curr_state,points_from_action,policy,probability,first_prob,first_policy=False)
            first_prob += first_prob_change
            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
                return states_dict
            curr_state = next_state

        long_distance = 0
        while selected_action[0][0] != "deposit" and long_distance < self.base_long_distance:
            next_state, selected_action = self.one_turn_no_policy(curr_state,points_from_action,first_prob=1)
            curr_state = next_state
            long_distance += 1
        self.deposite_dict(points_from_action, states_dict)
        return states_dict

    def deposit_sim_no_policy(self, first_prob=0.5, last_prob=0.9):
        curr_state = self.state_to_dict(self.initial)
        states_dict = {}
        points_from_action = []
        turns_to_go_change=[0,max(self.turns_to_go // 2, self.turns_to_go - 3 * self.base_long_distance, 1)]
        turns_to_go_change.append(max(self.turns_to_go - (self.base_long_distance // 2), self.turns_to_go - 10, turns_to_go_change[1] + 1))
        turns_to_go_change.append(max(self.turns_to_go, turns_to_go_change[2] + 1)),
        for i in range(turns_to_go_change[0], turns_to_go_change[1]):
            next_state, selected_action = self.one_turn_no_policy(curr_state,points_from_action,first_prob)
            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
            curr_state = next_state

        first_prob_change = (last_prob - first_prob) / (turns_to_go_change[2] - turns_to_go_change[1])
        first_prob += first_prob_change
        for i in range(turns_to_go_change[1], turns_to_go_change[2]):
            next_state, selected_action = self.one_turn_no_policy(curr_state,points_from_action,first_prob)
            first_prob += first_prob_change
            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
            curr_state = next_state

        first_prob_change = (last_prob - first_prob) / (turns_to_go_change[3] - turns_to_go_change[2])
        first_prob += first_prob_change
        for i in range(turns_to_go_change[2], turns_to_go_change[3]):
            next_state, selected_action = self.one_turn_no_policy(curr_state,points_from_action,first_prob)
            first_prob += first_prob_change
            if selected_action[0][0] == "deposit":
                self.deposite_dict(points_from_action, states_dict)
                return states_dict
            curr_state = next_state

        long_distance=0
        while selected_action[0][0] != "deposit" and long_distance < self.base_long_distance:
            next_state, selected_action = self.one_turn_no_policy(curr_state,points_from_action,first_prob=1)
            curr_state = next_state
            long_distance+=1
        self.deposite_dict(points_from_action, states_dict)
        return states_dict

    def policy_update(self, policy, state_and_actions_dict):
        for key, actions in state_and_actions_dict.items():
            policy[key] = {}
            max_ratio = float('-inf')
            best_action = None
            best_distance = float('inf')

            for action, info in actions.items():
                count = info['count']
                points = info['points']
                distance = info['distance']

                if count > 0:
                    ratio = points / count
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_action = action
                        best_distance = distance

            policy[key]['best_action'] = best_action
            policy[key]['max_points_count_ratio'] = max_ratio
            policy[key]['best_distance'] = best_distance

    def simulation(self, state_and_actions_dict, run_length, policy_parms=None):
        for i in range(run_length):
            if policy_parms:
                policy, policy_prob= policy_parms
                sim_result = self.deposit_sim_policy(policy, policy_prob)
            else:
                sim_result = self.deposit_sim_no_policy()

            for key, actions in sim_result.items():
                if key not in state_and_actions_dict:
                    state_and_actions_dict[key] = {}

                for action, info in actions.items():
                    if action not in state_and_actions_dict[key]:
                        state_and_actions_dict[key][action] = {'points': 0, 'count': 0, 'distance':0}
                    state_and_actions_dict[key][action]['count'] += info['count']
                    state_and_actions_dict[key][action]['points'] += info['points']
                    state_and_actions_dict[key][action]['distance'] += info['distance']

    def simulate_of_simulation(self, num_runs=1000, reps=15, first_prob=0.2, last_prob=0.85):
        policy = {}
        state_and_actions_dict = {}
        self.simulation(state_and_actions_dict, num_runs)
        self.policy_update(policy, state_and_actions_dict)

        for turn in range(reps):
            prob_change = first_prob + (last_prob - first_prob) * (turn) / (reps - 1)
            self.simulation(state_and_actions_dict, num_runs, (policy, prob_change))
            self.policy_update(policy, state_and_actions_dict)
            t2=time.time()
            if(t2-self.t>295):
                return policy

        return policy

    #BFS and map combinations functions:
    def max_comb_map(self, bfs_maps):
        first_map=bfs_maps[0]
        cols=len(first_map[0])
        rows=len(first_map)
        new_map=[[0 for m in range(cols)] for n in range(rows)]
        for n in range(rows):
            for m in range(cols):
                new_map[n][m] = max(bfs_maps, key=lambda x: x[n][m])[n][m]
        return new_map

    def treasures_prob_moving_matrix(self, treasures_num, probability):
        moving_probability = probability / treasures_num
        not_moving_probability = (1 - probability) + moving_probability
        prob_moving_matrix = [[moving_probability] * treasures_num for i in range(treasures_num)]
        for i in range(treasures_num):
            prob_moving_matrix[i][i] = not_moving_probability
        return prob_moving_matrix

    def BFS_map(self, first_location, probability=None):
        is_visit = set()
        flag = False
        fifo_q = utils.FIFOQueue()
        if probability is not None:
            prob_moving_change_matrix=[]
            flag = True
            prob_moving_matrix = self.treasures_prob_moving_matrix(probability[0], probability[1])

            for row in prob_moving_matrix:
                prob_moving_change_matrix.append(row.copy())

            change_dist=1

        dist = 1
        bfs_map=[[-1 for _ in range(len(row))] for row in self.map]
        bfs_map[first_location[0]][first_location[1]] = 1
        fifo_q.append((first_location, dist))
        is_visit.add(first_location)

        while len(fifo_q) > 0:
            location, dist = fifo_q.pop()
            locations = []
            if (location[0] > 0):
                locations.append((location[0] - 1, location[1]))
            if (location[1] > 0):
                locations.append((location[0], location[1] - 1))
            if (location[0] < len(self.map) - 1):
                locations.append((location[0] + 1, location[1]))
            if (location[1] < len(self.map[0]) - 1):
                locations.append((location[0], location[1] + 1))

            for loc in locations:
                if loc not in is_visit:
                    is_visit.add(loc)
                    cell_info = self.map[loc[0]][loc[1]]
                    if cell_info == "B" or cell_info == "S":
                        fifo_q.append((loc, dist + 1))
                        if flag:
                            while dist > change_dist:
                                prob_moving_change_matrix=utils.matrix_multiplication(prob_moving_change_matrix, prob_moving_matrix)
                                change_dist+=1
                            bfs_map[loc[0]][loc[1]]=prob_moving_change_matrix[0][0]
                        else:
                            bfs_map[loc[0]][loc[1]]=(1/dist)
        if not flag:
            self.base_long_distance = 2 * dist
        return bfs_map

    def create_BFS_comb_map(self, treasures):
        total_length=1
        names=[]
        lengths=[]
        for treasure in treasures:
            names.append(treasure)
            length=len(treasures[treasure]["possible_locations"])
            lengths.append(length)
            total_length *= length

        base_map=self.BFS_map(self.base_location)
        combinations={"base":base_map}
        num_of_treasures=len(names)
        for i in range(total_length):
            curr = i
            names_locations=utils.hashabledict()
            names_locations["base"]=False
            created_bfs_maps=[]

            for j in range(num_of_treasures):
                name=names[j]
                length=lengths[j]
                probability=treasures[name]["prob_change_location"]
                idx=curr % length
                possible_locations=treasures[name]["possible_locations"][idx]
                names_locations[name]=possible_locations
                created_bfs_maps.append(self.BFS_map(possible_locations, (length, probability)))
                curr =curr//length

            max_comb=self.max_comb_map(created_bfs_maps)
            combinations[names_locations]=max_comb
            names_locations = copy.deepcopy(names_locations)
            names_locations["base"]=True

            max_comb=[max_comb, base_map]
            combinations[names_locations] = self.max_comb_map(max_comb)

        return combinations

    def create_keys(self, t_dict, base_location):
        key_dict=utils.hashabledict()
        key_dict["base"]=base_location
        for key in t_dict:
            key_dict[key]=t_dict[key]["location"]
        return key_dict

#continue:
    def our_h(self, list_of_actions, base_location, t_dict):
        selected_action=None
        for action in list_of_actions:
            if action[0][0] == "collect":
                    selected_action = action
            elif selected_action == None and action[0][0] == "deposit":
                    selected_action= action

        if selected_action is None:
            if t_dict:
                key = self.create_keys(t_dict, base_location)
            else:
                key = "base"
            max_score=-1

            for action in list_of_actions:
                if action[0][0] == "sail":
                    score=self.BFS_comb_map[key][action[0][2][0]][action[0][2][1]]
                    if score > max_score:
                        max_score=score
                        selected_action=action

        return selected_action

    def __init__(self, initial):
        self.t=time.time()
        pirate_ships= initial["pirate_ships"]
        self.num_of_pirates=len(pirate_ships)
        self.ships_names=[key for key in pirate_ships.keys()]
        self.first_pirate_name=self.ships_names[0]
        self.initial = initial
        self.turns_to_go = initial["turns to go"]
        self.treasures_names = list(initial["treasures"].keys())
        self.marines_names = list(initial["marine_ships"].keys())
        self.base_location = next(iter(initial['pirate_ships'].values()), None).get("location")
        self.map = initial['map']
        self.grid_rows = len(self.map)
        self.grid_cols = len(self.map[0])
        self.base_long_distance=0
        self.BFS_comb_map= self.create_BFS_comb_map(self.initial["treasures"])
        self.policy = self.simulate_of_simulation()



    def act(self, state):
        dict_state=self.state_to_dict(state)
        state_key=json.dumps(dict_state, sort_keys=True)

        if state_key in self.policy:
            action=self.policy[state_key]["best_action"]

        else:
            capacity = dict_state["pirate_ships"][self.ships_names[0]]["capacity"]
            list_of_actions = self.actions_for_state(dict_state)
            if capacity == 1:
                action = self.our_h(list_of_actions, True, dict_state["treasures"])
            elif capacity == 2:
                action = self.our_h(list_of_actions, False, dict_state["treasures"])
            else:
                action = self.our_h(list_of_actions, True, False)

        action=self.one_pirate_instead_of_many(action)
        return action

    def one_pirate_instead_of_many(self, action):
        if (action == "terminate" or action == "reset"):
            return action

        action=action[0]
        if (action[0] == "collect" or action[0] == "sail"):
            return tuple([(action[0], self.ships_names[i], action[2]) for i in range(self.num_of_pirates)])

        return tuple([(action[0], self.ships_names[i]) for i in range(self.num_of_pirates)])

#________________________________________INFINITE_____________________________________________________________
class InfinitePirateAgent:

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.map), len(
            self.map[0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def state_to_dict(self, data_dict):
        new_data = {}
        keys_to_get = ["pirate_ships", "treasures", "marine_ships"]
        for key in keys_to_get:
            if key in data_dict:
                new_data[key] = data_dict[key]
        return new_data

    def create_all_states(self, original_state):
        our_map = original_state["map"]
        treasures = original_state["treasures"]
        pirates = original_state["pirate_ships"]
        marine_ships = original_state["marine_ships"]

        ships_can_be = []
        for row in range(len(our_map)):
            for col in range(len(our_map[0])):
                if our_map[row][col] != 'I':
                    ships_can_be.append((row, col))

        pirate_positions_comb = list(itertools.product(ships_can_be, repeat=len(pirates)))
        capacity_comb = []

        for pirate in pirates.values():
            capacity_comb.append(range(pirate["capacity"] + 1))

        possible_states = []
        treasure_positions_comb = itertools.product(
            *[treasure["possible_locations"] for treasure in treasures.values()])
        marine_indexes_combinations = itertools.product(
            *[range(len(navy_ship["path"])) for navy_ship in marine_ships.values()])

        for pirate_combination, capacity_combination, treasure_combination, marine_index_combination \
                in itertools.product(pirate_positions_comb, itertools.product(*capacity_comb), treasure_positions_comb,
                                     marine_indexes_combinations):
            new_state = copy.deepcopy(original_state)

            for (location, capacity), pirate_ship_name in zip(zip(pirate_combination, capacity_combination),
                                                              pirates.keys()):
                new_state["pirate_ships"][pirate_ship_name]["capacity"] = capacity
                new_state["pirate_ships"][pirate_ship_name]["location"] = location

            for idx, marine_ship_name in zip(marine_index_combination, marine_ships.keys()):
                new_state["marine_ships"][marine_ship_name]["index"] = idx

            for location, treasure_name in zip(treasure_combination, treasures.keys()):
                if (location in self.initial["treasures"][treasure_name]["possible_locations"]):
                    new_state["treasures"][treasure_name]["location"] = location

            possible_states.append(self.state_to_dict(new_state))

        return possible_states

    def create_all_combinations(self, current_state):
        combined_locations_and_probabilities = []
        for treasure_key, treasure_dict in current_state['treasures'].items():
            treasure_location_probability_pairs = []

            for location in treasure_dict['possible_locations']:
                treasure_probability = treasure_dict['prob_change_location']
                if location != treasure_dict['location']:
                    probability = treasure_probability / len(treasure_dict['possible_locations'])
                else:
                    probability = treasure_probability / len(treasure_dict['possible_locations']) + (
                                1 - treasure_probability)

                treasure_location_probability_pairs.append((treasure_key, location, probability))
            combined_locations_and_probabilities.append(treasure_location_probability_pairs)

            for marine_name, marine_dict in current_state['marine_ships'].items():
                marine_location_probability_pairs = []
                curr_index = marine_dict['index']
                if len(marine_dict['path']) == 1:
                    prob = 1
                    marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                elif (curr_index == 0):
                    prob = 1 / 2
                    marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                    marine_location_probability_pairs += [(marine_name, curr_index + 1, prob)]
                elif (curr_index == len(marine_dict['path']) - 1):
                    prob = 1 / 2
                    marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                    marine_location_probability_pairs += [(marine_name, curr_index - 1, prob)]
                else:
                    prob = 1 / 3
                    marine_location_probability_pairs += [(marine_name, curr_index, prob)]
                    marine_location_probability_pairs += [(marine_name, curr_index - 1, prob)]
                    marine_location_probability_pairs += [(marine_name, curr_index + 1, prob)]

                combined_locations_and_probabilities.append(marine_location_probability_pairs)

        combinations_all = list(itertools.product(*combined_locations_and_probabilities))

        final_combinations_all = []
        for comb in combinations_all:
            probabilities = [item[-1] for item in comb]
            product_of_probabilities = 1.0
            for prob in probabilities:
                product_of_probabilities *= prob
            result = (product_of_probabilities,) + comb
            final_combinations_all.append(result)
        return final_combinations_all

    def all_actions_for_state(self, state):
        all_ship_actions = []
        pirate_ships = state["pirate_ships"]
        all_treasures = state["treasures"]

        for pirate_ship_name, pirate_ship_dict in pirate_ships.items():
            actions_per_ship = []

            cord_x = pirate_ship_dict["location"][0]
            cord_y = pirate_ship_dict["location"][1]
            possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1),
                              (cord_x, cord_y - 1)]

            # sail
            for move in possible_moves:
                if self.is_valid_sail(move[0], move[1]):
                    actions_per_ship += [("sail", pirate_ship_name, move)]

            # collect
            if pirate_ship_dict["capacity"] > 0:
                near_islands_list = []
                for move in possible_moves:
                    if self.is_adj_island(move[0], move[1]):
                        near_islands_list.append((move[0], move[1]))

                for treasure_name, treasure_dict in all_treasures.items():
                    if treasure_dict["location"] in near_islands_list:
                        actions_per_ship += [("collect", pirate_ship_name, treasure_name)]

            # deposit
            if (self.is_valid_deposit(cord_x, cord_y) and pirate_ship_dict['capacity'] <
                    self.initial['pirate_ships'][pirate_ship_name]['capacity']):
                actions_per_ship += [("deposit", pirate_ship_name)]

            # wait
            actions_per_ship += [("wait", pirate_ship_name)]

            all_ship_actions.append(actions_per_ship)

        all_combinations = list(itertools.product(*all_ship_actions))
        all_combinations += ["reset"]
        # all_combinations += ["terminate"]
        return all_combinations

    def is_valid_sail(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] in ['S', 'B']):
            return True
        return False

    def is_valid_deposit(self, x, y):
        if (self.map[x][y] == 'B'):
            return True
        return False

    def is_adj_island(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] == 'I'):
            return True
        return False

    def next_state_prob_and_points(self, current_state, action):
        new_state_initial = copy.deepcopy(current_state)

        #reset
        if action == "reset":
            new_state = copy.deepcopy(self.initial)
            point_to_change = -2

            marine_ship_locations_list = []
            for marine in new_state['marine_ships'].values():
                marine_ship_locations_list.append(marine['path'][marine['index']])

            for ship_name, ship in new_state['pirate_ships'].items():
                if ship['location'] in marine_ship_locations_list:
                    point_to_change -= 1
                    new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name]['capacity']

            return [(self.state_to_dict(new_state), 1, point_to_change)]

        #all other actions:
        else:
            for current_action in action:
                if current_action[0] == 'sail':
                    new_state_initial['pirate_ships'][current_action[1]]['location'] = current_action[2]

                if current_action[0] == 'collect':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] -= 1

                if current_action[0] == 'deposit':
                    new_state_initial['pirate_ships'][current_action[1]]['capacity'] = self.initial['pirate_ships'][current_action[1]]['capacity']
            combinations = self.create_all_combinations(current_state)

            next_state_prob_and_points_list = []
            for comb in combinations:
                new_state = copy.deepcopy(new_state_initial)
                for item in comb[1:]:
                    if item[0] in self.treasures_names:
                        new_state['treasures'][item[0]]['location'] = item[1]
                    else:
                        if item[0] in self.marines_names:
                            new_state['marine_ships'][item[0]]['index'] = item[1]

                marine_distances_and_locations = []
                for marine_item in new_state['marine_ships'].values():
                    marine_distances_and_locations.append(marine_item['path'][marine_item['index']])

                point_to_change = 0
                for current_action in action:
                    if current_action[0] == "deposit":
                        point_to_change += 4 * (self.initial['pirate_ships'][current_action[1]]['capacity'] - current_state['pirate_ships'][current_action[1]]['capacity'])

                for pirate_name, pirate_dict in new_state['pirate_ships'].items():
                    if pirate_dict['location'] in marine_distances_and_locations:
                        point_to_change -= 1
                        new_state['pirate_ships'][pirate_name]['capacity'] = self.initial['pirate_ships'][pirate_name]['capacity']

                next_state_prob_and_points_list += [(self.state_to_dict(new_state), comb[0], point_to_change)]

            return next_state_prob_and_points_list

    def VI_infinite(self):
        old_values = {}
        for state in self.states:
            old_values[json.dumps(state)] = 0

        current_values = {}
        for state in self.states:
            current_values[json.dumps(state)] = 0

        all_states_actions = {}
        for state in self.states:
            all_states_actions[json.dumps(state)] = self.all_actions_for_state(state)

        next_states = {}
        for state in self.states:
            state_actions = self.all_actions_for_state(state)
            action_probs = {}
            for action in state_actions:
                action_probs[action] = []
            next_states[json.dumps(state)] = action_probs

        for state in self.states:
            for action in self.all_actions_for_state(state):
                next_states[json.dumps(state)][action] = self.next_state_prob_and_points(state, action)

        best_to_do = {}
        for state in self.states:
            best_to_do[json.dumps(state)] = 0

        converged = False

        while not converged:
            converged = True
            for state in self.states:
                max_value = float('-inf')
                best_action = None
                actions_from_state = all_states_actions[json.dumps(state)]
                for action in actions_from_state:
                    action_value = 0
                    for next_state, probability, points_to_change in next_states[json.dumps(state)][action]:
                        action_value += probability * (self.gamma*(old_values[json.dumps(next_state)]) + points_to_change)
                    if action_value > max_value:
                        max_value = action_value
                        best_action = action

                current_values[json.dumps(state)] = max_value
                best_to_do[json.dumps(state)] = best_action

                if abs(current_values[json.dumps(state)] - old_values[json.dumps(state)]) > 0.01:
                    converged = False

            old_values = current_values.copy()

        return best_to_do, current_values

    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma
        self.curr_time = time.time()
        self.marines_names = list(initial["marine_ships"].keys())
        self.treasures_names = list(initial["treasures"].keys())
        self.base_location = next(iter(initial['pirate_ships'].values()), None).get("location")
        self.map = initial['map']
        self.grid_rows = len(self.map)
        self.grid_cols = len(self.map[0])
        self.turn_now = 0
        self.graph = self.build_graph()
        self.states = self.create_all_states(self.initial)
        self.policy, self.VI_infinite_value = self.VI_infinite()

    def act(self, state):
        state = self.state_to_dict(state)
        return self.policy[json.dumps(state)]

    def value(self, state):
        return self.VI_infinite_value[json.dumps(self.state_to_dict(state))]