import search_algorithm
import json
import itertools
import copy

UNCOLLECTED = 0
ON_THE_WAY = 1
DEPOSIT = 2


class OnePieceProblem(search_algorithm.Problem):
    """This class implements a medical problem according to problem description file"""
    #-----------------------init-----------------------
    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial.pop("map")
        self.grid_rows = len(self.map)
        self.grid_cols = len(self.map[0])

        pirate_ships = initial["pirate_ships"]
        for ship_name, ship_dict in pirate_ships.items():
            self.base = ship_dict
            break

        search_algorithm.Problem.__init__(self, dict_to_initial_state(initial))
        self.distances_for_h_2 = bfs_all_distances(self.map)
        self.distances_for_h_3 = bfs_consider_island(self.map)
        self.dist_matrix = self.bfs_between_every_two_slots(self.map)

#---------------------our functions--------------
    def is_valid_sail(self, x, y):
        if (x >= 0 and y >= 0) and (x < self.grid_rows and y < self.grid_cols) and (self.map[x][y] in ['S','B']):
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

    def next_positions_after_move_marines(self, marines):
        next_positions = []
        for marine in marines.values():
            patrol_route = marine["track"]
            route_length = len(patrol_route)
            current_position_index = marine["location_index"]
            next_position_index = (current_position_index + 1) % route_length
            next_positions.append(patrol_route[next_position_index])
        return next_positions

    def who_is_my_sea_adj(self, cord_x, cord_y, distances):
        possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1), (cord_x, cord_y - 1)]
        my_sea_adj_distances = []
        for move in possible_moves:
            x, y = move
            if (0 <= x < self.grid_rows) and (0 <= y < self.grid_cols) and (self.map[x][y] == 'S'):
                my_sea_adj_distances.append(distances[x][y])

        return my_sea_adj_distances

    def who_is_my_sea_adj_with_B(self, cord_x, cord_y, distances):
        possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1), (cord_x, cord_y - 1)]
        my_sea_adj_distances = []
        for move in possible_moves:
            x, y = move
            if (0 <= x < self.grid_rows) and (0 <= y < self.grid_cols) and (self.map[x][y] in ['S', 'B']):
                my_sea_adj_distances.append(distances[x][y])

        return my_sea_adj_distances

    def find_min_in_list(self,my_list):
        min= float('inf')
        for i in my_list:
            if i>=0 and i<min:
                min=i

        return min

#-------------------------actions------------------------
    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        state = state_to_dict(state)
        actions_per_pirate_ship = {}
        pirate_ships = state["pirate_ships"]
        all_treasures = state["treasures"]
        marine_ships=state["marine_ships"]
        marines_after_this_turn_location = self.next_positions_after_move_marines(marine_ships)

        for pirate_ship_name, pirate_ship_dict in pirate_ships.items():

            cord_x = pirate_ship_dict["location"][0]
            cord_y = pirate_ship_dict["location"][1]
            possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1),
                                  (cord_x, cord_y - 1)]

            # wait
            actions_per_pirate_ship[pirate_ship_name] = [("wait", pirate_ship_name)]

            # sail
            for move in possible_moves:
                if self.is_valid_sail(move[0], move[1]) and\
                        ((move[0],move[1]) not in marines_after_this_turn_location or pirate_ship_dict["treasures_num"]==0):
                    actions_per_pirate_ship[pirate_ship_name].append(("sail", pirate_ship_name, move))

            #collect
            if pirate_ship_dict["treasures_num"] <2:
                near_islands_list=[]
                for move in possible_moves:
                    if self.is_adj_island(move[0], move[1]):
                        near_islands_list.append([move[0], move[1]])

                for treasure_name, treasure_dict in all_treasures.items():
                    if treasure_dict["location"] in near_islands_list and treasure_dict["is_picked"] != DEPOSIT and \
                            treasure_name not in pirate_ship_dict["treasures_names"] :
                        if not (pirate_ship_name != max(pirate_ships.keys()) and treasure_name == min(all_treasures.keys())):
                            if (cord_x, cord_y) not in marines_after_this_turn_location:
                                actions_per_pirate_ship[pirate_ship_name].append(("collect_treasure", pirate_ship_name, treasure_name))

            # deposit
            if(self.is_valid_deposit(cord_x,cord_y) and
                    (pirate_ship_dict["treasures_num"]==1 or pirate_ship_dict["treasures_num"]==2 )):
                    actions_per_pirate_ship[pirate_ship_name].append(("deposit_treasures", pirate_ship_name))

        actions = tuple(itertools.product(*list(actions_per_pirate_ship.values())))
        return actions

#---------------------------------------------result-------------------
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = self.result_for_pirate(state, action)
        new_state2 = self.result_for_marines(new_state)
        new_state3 = self.update_treasures(new_state2)
        return new_state3

    def result_for_pirate(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = state_to_dict(state)
        next_state = copy.deepcopy(state)
        for sub_action in action:
            action_name = sub_action[0]
            pirate_ship_name = sub_action[1]

            # WAIT
            if action_name == "wait":
                next_state = dict_to_state(state)
                return next_state

            #SAIL
            elif action_name == "sail":
                new_location = list(sub_action[2])
                next_state["pirate_ships"][pirate_ship_name]["location"] = new_location

            #COLLECT
            elif action_name == "collect_treasure":
                treasure_name = sub_action[2]
                #what i do to the ship:
                next_state["pirate_ships"][pirate_ship_name]["treasures_num"] += 1
                next_state["pirate_ships"][pirate_ship_name]["treasures_names"].append(treasure_name)
                next_state["treasures"][treasure_name]["ships_collect_me_names"].append(pirate_ship_name)
                #what i do to the treasure:
                if(next_state["treasures"][treasure_name]["is_picked"] == UNCOLLECTED):
                    next_state["treasures"][treasure_name]["is_picked"] = ON_THE_WAY
                    next_state["uncollected_treasures_num"] -= 1
                next_state["treasures"][treasure_name]["ships_collect_me_names"].sort()
                next_state["pirate_ships"][pirate_ship_name]["treasures_names"].sort()

            #DEPOSIT
            elif action_name == "deposit_treasures":
                my_treasures_list= state["pirate_ships"][pirate_ship_name]["treasures_names"]
                for treasure in my_treasures_list:
                    if(next_state["treasures"][treasure]["is_picked"] == DEPOSIT):
                        next_state["pirate_ships"][pirate_ship_name]["treasures_num"] -= 1
                        next_state["pirate_ships"][pirate_ship_name]["treasures_names"].remove(treasure)
                        next_state["treasures"][treasure]["ships_collect_me_names"].remove(pirate_ship_name)
                    else:
                        next_state["pirate_ships"][pirate_ship_name]["treasures_num"] -= 1
                        next_state["pirate_ships"][pirate_ship_name]["treasures_names"].remove(treasure)
                        next_state["treasures"][treasure]["is_picked"] = DEPOSIT
                        next_state["not_deposit_num"] -= 1
                        next_state["treasures"][treasure]["ships_collect_me_names"].remove(pirate_ship_name)

        next_state = dict_to_state(next_state)
        return next_state

    def result_for_marines(self, state):
        state_dict = state_to_dict(state)
        marine_ships = state_dict["marine_ships"]
        for marine_ship_name, marine_ship_info in marine_ships.items():
            track = marine_ship_info["track"]
            track_length = len(track)
            current_index = marine_ship_info["location_index"]
            marine_ship_info["location_index"] = (current_index + 1) % track_length
        next_state = dict_to_state(state_dict)
        return next_state

    def update_treasures(self, state):
        state_dict = state_to_dict(state)
        marines = state_dict["marine_ships"]
        marines_locations = []
        for marine_name, marine_dict in marines.items():
            path = marine_dict["track"]
            index = marine_dict["location_index"]
            marines_locations.append(path[index])

        pirate_ships = state_dict["pirate_ships"]
        for pirate_name, pirate_dict in pirate_ships.items():
            location = pirate_dict["location"]
            if location in marines_locations and  pirate_dict["treasures_num"]>0:
                for treasure in pirate_ships[pirate_name]["treasures_names"]:
                    state_dict["treasures"][treasure]["ships_collect_me_names"].remove(pirate_name)
                    if(len(state_dict["treasures"][treasure]["ships_collect_me_names"])==0 and
                            state_dict["treasures"][treasure]["is_picked"] != DEPOSIT):
                        state_dict["treasures"][treasure]["is_picked"] = UNCOLLECTED
                        state_dict["uncollected_treasures_num"] += 1
                #remove treasures from ships
                pirate_dict["treasures_names"] = []
                pirate_dict["treasures_num"]=0

        next_state = dict_to_state(state_dict)
        return next_state

#-----goal test----------------
    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        st = state_to_dict(state)
        if st["not_deposit_num"] == 0:
            return True
        return False


#---------------------our H's----------------------------------------
    def h(self, node):
        dict = state_to_dict(node.state)
        if len(dict["pirate_ships"])<=2:
            return max(self.h_3(node) , self.h_1(node)) #they are both admissible
        else:
            return self.h_4(node) #thats run faster for more than 2 ships

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = state_to_dict(node.state)
        uncollected_treasures_num = state["uncollected_treasures_num"]
        pirate_ships_num = len(state["pirate_ships"])
        return uncollected_treasures_num / pirate_ships_num

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def h_2(self, node):
        state = state_to_dict(node.state)
        treasures = state["treasures"]
        distances = self.distances_for_h_2
        sum_of_distances = 0
        for treasure_name, treasure_dict in treasures.items():
            initial_island_location = treasure_dict["location"]
            if treasure_dict["is_picked"] == UNCOLLECTED:
                my_adj_1 = self.who_is_my_sea_adj(initial_island_location[0], initial_island_location[1], distances)
                min_local_1 = self.find_min_in_list(my_adj_1)
                sum_of_distances += min_local_1

            elif treasure_dict["is_picked"] == ON_THE_WAY and treasure_dict["ships_collect_me_names"] != []:
                pirate_ships = state["pirate_ships"]
                ships_collect_me = treasure_dict["ships_collect_me_names"]
                min_for_all_the_ships = float('inf')

                for pirate_name, pirate_dict in pirate_ships.items():
                    if pirate_name in ships_collect_me:
                        ship_location_x = pirate_dict["location"][0]
                        ship_location_y = pirate_dict["location"][1]
                        my_adj_2 = self.who_is_my_sea_adj(ship_location_x, ship_location_y, distances)
                        min_local_2 = self.find_min_in_list(my_adj_2)
                        if (min_local_2 < min_for_all_the_ships):
                            min_for_all_the_ships = min_local_2

                sum_of_distances += min_for_all_the_ships

        pirate_ships_num = len(state["pirate_ships"])
        return sum_of_distances / pirate_ships_num

    def h_3(self,node):
        state = state_to_dict(node.state)
        treasures = state["treasures"]
        distances = self.distances_for_h_3
        sum_of_distances=0
        for treasure_name, treasure_dict in treasures.items():
            initial_island_location = treasure_dict["location"]
            if treasure_dict["is_picked"] ==UNCOLLECTED:
                sum_of_distances+=distances[initial_island_location[0]][initial_island_location[1]]

            elif treasure_dict["is_picked"] == ON_THE_WAY and treasure_dict["ships_collect_me_names"]!=[] :
                pirate_ships = state["pirate_ships"]
                ships_collect_me=treasure_dict["ships_collect_me_names"]
                min_for_all_the_ships= float('inf')

                for pirate_name, pirate_dict in pirate_ships.items():
                    if pirate_name in ships_collect_me:
                        ship_location_x=pirate_dict["location"][0]
                        ship_location_y = pirate_dict["location"][1]
                        min_local_2=distances[ship_location_x][ship_location_y]
                        if(min_local_2<min_for_all_the_ships):
                            min_for_all_the_ships=min_local_2

                sum_of_distances+=min_for_all_the_ships

        pirate_ships_num = len(state["pirate_ships"])
        return sum_of_distances/pirate_ships_num

    def h_4(self, node):
        dict = state_to_dict(node.state)
        pirate_ships = dict["pirate_ships"]
        treasures = dict["treasures"]
        distances_to_return = 0
        used_treasures = []

        for pirate_name, pirate_dict in pirate_ships.items():
            pirate_cord_x, pirate_cord_y = pirate_dict["location"]
            if pirate_dict["treasures_num"] == 2:
                distances_to_return+= self.distances_for_h_3[pirate_cord_x][pirate_cord_y]
            elif pirate_dict["treasures_num"] == 1:
                sorted_treasures_h4 = self.sort_treasures(pirate_dict, treasures)
                IS_USED = False
                for treasure_name_sorted, treasure_sorted_dist in sorted_treasures_h4:
                    the_closest_ship, treasure_sorted_dist = self.find_closest_ship(treasure_name_sorted, pirate_ships, treasures)
                    if treasure_name_sorted not in used_treasures and the_closest_ship == pirate_name:
                        used_treasures.append(treasure_name_sorted)
                        IS_USED = True
                        distances_to_return+= treasure_sorted_dist
                        break
                if IS_USED == False:
                    distances_to_return+=self.distances_for_h_3[pirate_cord_x][pirate_cord_y]
            else:
                treasure_name_sorted, dist = self.find_closest_treasure(pirate_dict, treasures, used_treasures)

                if treasure_name_sorted == None:
                    continue
                used_treasures.append(treasure_name_sorted)
                distances_to_return += dist

        uncollected = self.uncollected_treasures(treasures)
        for treasure_name_sorted in uncollected:
            treasure_dict = treasures[treasure_name_sorted]
            if treasure_name_sorted is not None and treasure_name_sorted not in used_treasures:
                treasure_cord_x, treasure_cord_y = treasure_dict["location"]
                distance_from_neighbors_to_base= self.who_is_my_sea_adj_with_B(treasure_cord_x,treasure_cord_y,self.distances_for_h_3)
                distances_to_return+= 2*min(distance_from_neighbors_to_base, default=0)

        return distances_to_return

    def uncollected_treasures(self, treasures):
        list_to_return = []
        for treasure_name, treasure_dict in treasures.items():
            if treasure_dict["is_picked"] == UNCOLLECTED:
                list_to_return.append(treasure_name)
        return list_to_return

    def find_closest_ship(self, treasure_name, pirates, treasures):
        minimum_distance = float("inf")
        closest_ship_name = None
        treasure_cord_x, treasure_cord_y = treasures[treasure_name]["location"]
        for pirate_name, pirate_dict in pirates.items():
            if pirate_dict["treasures_num"]<2:
                pirate_cord_x, pirate_cord_y = pirate_dict["location"]
                minimum_distance_to_ship = min(self.who_is_my_sea_adj_for_h(treasure_cord_x, treasure_cord_y, pirate_cord_x, pirate_cord_y,
                                                                self.distances_for_h_3, self.dist_matrix), default=float("inf"))

                if minimum_distance_to_ship<minimum_distance:
                    minimum_distance = minimum_distance_to_ship
                    closest_ship_name = pirate_name

        return (closest_ship_name, minimum_distance)

    def find_closest_treasure(self, pirate_dict, treasures, used_treasures):
        minimum_distance = float("inf")
        pirate_cord_x, pirate_cord_y = pirate_dict["location"]
        closest_treasure_name = None
        for treasure_name in self.uncollected_treasures(treasures):
            treasure_dict = treasures[treasure_name]
            treasure_cord_x, treasure_cord_y = treasure_dict["location"]
            if treasure_name not in used_treasures:

                minimoum_distance_to_treasure = min(self.who_is_my_sea_adj_for_h(treasure_cord_x, treasure_cord_y,pirate_cord_x, pirate_cord_y, self.distances_for_h_3,
                                                                    self.dist_matrix), default=float("inf"))
                if minimoum_distance_to_treasure<minimum_distance:
                    minimum_distance = minimoum_distance_to_treasure
                    closest_treasure_name = treasure_name
        return (closest_treasure_name, minimum_distance)


    def sort_treasures(self, pirate_dict, treasures):
        x_cord_pirate, y_cord_pirate = pirate_dict["location"]
        dist_arr = []
        for treasure_name, treasure_dict in treasures.items():
            if treasure_dict["is_picked"] == UNCOLLECTED:
                treasure_x_cord, treasure_y_cord = treasures[treasure_name]["location"]
                min_local_3=min(self.who_is_my_sea_adj_for_h(treasure_x_cord, treasure_y_cord, x_cord_pirate, y_cord_pirate,
                                                             self.distances_for_h_3, self.dist_matrix), default=float("inf"))
                dist_arr.append((treasure_name, min_local_3))

        sorted_list_of_treasures = sorted(dist_arr, key=lambda x: x[1])
        return sorted_list_of_treasures


    def bfs_between_every_two_slots (self, grid):
        grid_height, grid_width = len(grid), len(grid[0])
        distance_grid = [
            [
                [
                    [-1 for _ in range(grid_width)]
                    for _ in range(grid_height)
                ]
                for _ in range(grid_width)
            ]
            for _ in range(grid_height)
        ]

        for origin_row in range(grid_height):
            for origin_col in range(grid_width):
                if grid[origin_row][origin_col] in ['S', 'B']:
                    exploration_queue = [(origin_row, origin_col, 0)]

                    has_visited = [[False] * grid_width for _ in range(grid_height)]
                    has_visited[origin_row][origin_col] = True

                    while exploration_queue:
                        current_row, current_col, current_distance = exploration_queue.pop(0)
                        distance_grid[origin_row][origin_col][current_row][current_col] = current_distance

                        for delta_row, delta_col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            next_row, next_col = current_row + delta_row, current_col + delta_col
                            if 0 <= next_row < grid_height and 0 <= next_col < grid_width and grid[next_row][
                                next_col] in ['S', 'B'] and not has_visited[next_row][next_col]:
                                has_visited[next_row][next_col] = True
                                exploration_queue.append((next_row, next_col, current_distance + 1))

        return distance_grid

    def who_is_my_sea_adj_for_h(self, cord_x, cord_y,x_cord_pirate, y_cord_pirate, distances, dist_mat):
        possible_moves = [(cord_x + 1, cord_y), (cord_x - 1, cord_y), (cord_x, cord_y + 1), (cord_x, cord_y - 1)]
        my_sea_adj_distances = []
        for move in possible_moves:
            x, y = move
            if (0 <= x < self.grid_rows) and (0 <= y < self.grid_cols) and (self.map[x][y] in ['S','B']):
                my_sea_adj_distances.append(distances[x][y]+dist_mat[x_cord_pirate][y_cord_pirate][x][y])
        return my_sea_adj_distances


#-----------problem----------------
def create_onepiece_problem(game):
    return OnePieceProblem(game)

def dict_to_initial_state(initial_dict):
    pirate_ships = initial_dict["pirate_ships"]
    treasures = initial_dict["treasures"]
    marine_ships = initial_dict["marine_ships"]

    initial_dict["not_deposit_num"] = len(treasures)
    initial_dict["uncollected_treasures_num"] = len(treasures)

    for pirate_ship_name, pirate_ship_location in pirate_ships.items():
        pirate_ships[pirate_ship_name] = {
            "treasures_num": 0,
            "treasures_names": [],
            "location": pirate_ship_location
        }

    for treasure_name, treasure_location in treasures.items():
        treasures[treasure_name] = {
            "is_picked": UNCOLLECTED,
            "location": treasure_location,
            "ships_collect_me_names": []
        }

    for marine_name, marine_track in marine_ships.items():
        marine_ships[marine_name] = {"track": marine_track + marine_track[::-1][1:-1], "location_index": 0}

    return json.dumps(initial_dict)

def state_to_dict(state):
    return json.loads(state)

def dict_to_state(dictionary):
    if dictionary is not None:
        return json.dumps(dictionary, sort_keys=True)
    else:
        return '{}'

#--------------BFS--------------------


def is_valid_in_grid(x, y, grid, visited):
    if (x >= 0 and y >= 0) and (x < len(grid) and y < len(grid[0])) and (visited[x][y] == False):
        return True
    return False


class QItem:
    def __init__(self, row, col, dist):
        self.row = row
        self.col = col
        self.dist = dist

    def __repr__(self):
        return f"QItem({self.row}, {self.col}, {self.dist})"

def bfs_all_distances(grid):
    source = QItem(0, 0, 0)

    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == 'B':
                source.row = row
                source.col = col
                break

    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    distance_grid = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]  # For storing distances

    queue = []
    queue.append(source)
    visited[source.row][source.col] = True
    distance_grid[source.row][source.col] = 0

    while len(queue) != 0:
        source = queue.pop(0)
        # moving up
        if is_valid_in_grid(source.row - 1, source.col, grid, visited):
            queue.append(QItem(source.row - 1, source.col, source.dist + 1))
            visited[source.row - 1][source.col] = True
            distance_grid[source.row - 1][source.col] = source.dist + 1
        # moving down
        if is_valid_in_grid(source.row + 1, source.col, grid, visited):
            queue.append(QItem(source.row + 1, source.col, source.dist + 1))
            visited[source.row + 1][source.col] = True
            distance_grid[source.row + 1][source.col] = source.dist + 1
        # moving left
        if is_valid_in_grid(source.row, source.col - 1, grid, visited):
            queue.append(QItem(source.row, source.col - 1, source.dist + 1))
            visited[source.row][source.col - 1] = True
            distance_grid[source.row][source.col - 1] = source.dist + 1
        # moving right
        if is_valid_in_grid(source.row, source.col + 1, grid, visited):
            queue.append(QItem(source.row, source.col + 1, source.dist + 1))
            visited[source.row][source.col + 1] = True
            distance_grid[source.row][source.col + 1] = source.dist + 1
    return distance_grid

def is_valid_in_grid_consider_island(x, y, grid, visited):
    if (x >= 0 and y >= 0) and (x < len(grid) and y < len(grid[0])) and (grid[x][y]!='I') and (visited[x][y] == False):
        return True
    return False

def bfs_consider_island(grid):
    source = QItem(0, 0, 0)

    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == 'B':
                source.row = row
                source.col = col
                break

    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    distance_grid = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]

    queue = []
    queue.append(source)
    visited[source.row][source.col] = True
    distance_grid[source.row][source.col] = 0

    while len(queue) != 0:
        source = queue.pop(0)
        # moving up
        if is_valid_in_grid_consider_island(source.row - 1, source.col, grid, visited):
            queue.append(QItem(source.row - 1, source.col, source.dist + 1))
            visited[source.row - 1][source.col] = True
            distance_grid[source.row - 1][source.col] = source.dist + 1
        # moving down
        if is_valid_in_grid_consider_island(source.row + 1, source.col, grid, visited):
            queue.append(QItem(source.row + 1, source.col, source.dist + 1))
            visited[source.row + 1][source.col] = True
            distance_grid[source.row + 1][source.col] = source.dist + 1
        # moving left
        if is_valid_in_grid_consider_island(source.row, source.col - 1, grid, visited):
            queue.append(QItem(source.row, source.col - 1, source.dist + 1))
            visited[source.row][source.col - 1] = True
            distance_grid[source.row][source.col - 1] = source.dist + 1
        # moving right
        if is_valid_in_grid_consider_island(source.row, source.col + 1, grid, visited):
            queue.append(QItem(source.row, source.col + 1, source.dist + 1))
            visited[source.row][source.col + 1] = True
            distance_grid[source.row][source.col + 1] = source.dist + 1
    return distance_grid







