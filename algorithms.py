import math
import time
import networkx as nx
import numpy as np
import kleindessner_etal_algorithms as kl
from multiprocessing import Process, Queue

curr_id = -1
data_file = None
group_file = None
att1 = -1
att2 = -1
dataset = ''
num_recs_to_read = -1
rank = -1
print_logs = False
class Point:
    def __init__(self, id, coordinates, group):
        self.id = id
        self.coordinates = coordinates
        self.group = group


def get_distance(point1, point2):
    global dataset
    dist = 0
    if dataset == 'sushia':
        for i in range(10):
            for j in range(i + 1, 10):
                point1_list = list(point1.coordinates)
                point2_list = list(point2.coordinates)
                ord1 = point1_list.index(i) - point1_list.index(j)
                ord2 = point2_list.index(i) - point2_list.index(j)
                if ord1 * ord2 < 0:
                    dist += 1
        return dist
    if dataset == 'celeba' or dataset == 'sushib' or dataset == 'adult' or dataset == 'random_euclidean':
        return np.linalg.norm(point1.coordinates - point2.coordinates, ord=1)


# Do it only after get_next_point() returns None
def start_new_pass():
    global dataset
    global data_file
    global data_file_name
    global group_file_name
    global group_file
    global curr_id
    data_file = open(data_file_name, 'r')
    if dataset == 'sushia':
        data_file.readline() #first line is metadata
    group_file = open(group_file_name, 'r')
    if dataset == 'celeba':
        group_file.readline() #first line is number of records
        group_file.readline() #second line is names of attributes
    curr_id = -1

def end_pass():
    global data_file
    global group_file
    global curr_id
    curr_id = -1
    if not data_file.closed:
        data_file.close()
    if not group_file.closed:
        group_file.close()


# 1 5_o_Clock_Shadow
# 2 Arched_Eyebrows
# 3 Attractive
# 4 Bags_Under_Eyes
# 5 Bald
# 6 Bangs
# 7 Big_Lips
# 8 Big_Nose
# 9 Black_Hair
# 10 Blond_Hair
# 11 Blurry
# 12 Brown_Hair
# 13 Bushy_Eyebrows
# 14 Chubby
# 15 Double_Chin
# 16 Eyeglasses
# 17 Goatee
# 18 Gray_Hair
# 19 Heavy_Makeup
# 20 High_Cheekbones
# 21 Male
# 22 Mouth_Slightly_Open
# 23 Mustache
# 24 Narrow_Eyes
# 25 No_Beard
# 26 Oval_Face
# 27 Pale_Skin
# 28 Pointy_Nose
# 29 Receding_Hairline
# 30 Rosy_Cheeks
# 31 Sideburns
# 32 Smiling
# 33 Straight_Hair
# 34 Wavy_Hair
# 35 Wearing_Earrings
# 36 Wearing_Hat
# 37 Wearing_Lipstick
# 38 Wearing_Necklace
# 39 Wearing_Necktie
# 40 Young


def get_next_point():
    global dataset
    global data_file
    global group_file
    global curr_id
    global att1, att2
    global num_recs_to_read
    global capacities
    curr_id += 1
    data_line = data_file.readline()
    if len(data_line) == 0 or curr_id == num_recs_to_read:
        end_pass()
        return None
    if dataset == 'celeba':
        coordinates = np.fromstring(data_line, sep = ',')
        group_line = group_file.readline()
        attr1 = group_line.split()[att1]
        if att2 == -1:
            if attr1 == '1':
                group = 0
            else:
                group = 1
        else:
            attr2 = group_line.split()[att2]
            if attr1 == '1':
                if attr2 == '1':
                    group = 0
                else:
                    group = 1
            else:
                if attr2 == '1':
                    group = 2
                else:
                    group = 3
    if dataset == 'sushia' or dataset == 'sushib':
        coordinates = np.fromstring(data_line, sep = ' ')[2:]
        attr_list = group_file.readline().split('\t')
        gender = int(attr_list[1])
        age_group = int(attr_list[2])
        if len(capacities) == 12:
            group = 6*gender + age_group
        if len(capacities) == 6:
            group = age_group
        if len(capacities) == 2:
            group = gender

    if dataset == 'adult':
        coordinates = np.fromstring(data_line, sep = ',')
        attr_list = group_file.readline().split(', ')
        if attr_list[9] == 'Male':
            gender = 0
        else:
            gender = 1
        race_list = ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
        race = race_list.index(attr_list[8])
        if len(capacities) == 10:
            group = 5*gender + race
        if len(capacities) == 5:
            group = race
        if len(capacities) == 2:
            group = gender

    if dataset == 'random_euclidean':
        coordinates = np.fromstring(data_line, sep=',')
        group = int(group_file.readline().strip())
    return Point(curr_id, coordinates, group)


def compute_cost(solution):
    cost = 0.0
    start_new_pass()
    point = get_next_point()
    while point is not None:
        dist = float("inf")
        for solp in solution:
            dist = min(dist, get_distance(solp, point))
        cost = max(cost, dist)
        point = get_next_point()
    return cost


def get_distance_points(i, points):
    dist = float("inf")
    cert = -1
    for point in points:
        cur_dist = get_distance(i, point)
        if cur_dist < dist:
            dist = cur_dist
            cert = point
    return dist, cert


def get_simple_opt_lower_bound():
    global rank
    points = []
    start_new_pass()
    for i in range(rank + 1):
        points.append(get_next_point())
    end_pass()
    min_dist = float("inf")
    for i in points:
        dist_i = float("inf")
        for j in points:
            cur_dist = get_distance(i,j)
            if j != i and cur_dist < dist_i:
                dist_i = cur_dist
        if dist_i < min_dist:
            min_dist = dist_i
    return min_dist / 2.0


def get_arb_solution():
    arb_solution = []
    global capacities
    residual_capacities = capacities.copy()
    total_residual_capacity = sum(residual_capacities)
    start_new_pass()
    point = get_next_point()
    while point is not None:
        if residual_capacities[point.group] > 0:
            arb_solution.append(point)
            residual_capacities[point.group] -= 1
            total_residual_capacity -= 1
            if total_residual_capacity == 0:
                end_pass()
                return arb_solution
        point = get_next_point()
    return []

def get_arb_solution_from_points(points):
    arb_solution = []
    global capacities
    residual_capacities = capacities.copy()
    total_residual_capacity = sum(residual_capacities)
    for point in points:
        if residual_capacities[point.group] > 0:
            arb_solution.append(point)
            residual_capacities[point.group] -= 1
            total_residual_capacity -= 1
            if total_residual_capacity == 0:
                end_pass()
                return arb_solution
    return []

def get_opt_lower_bound():
   return get_unfair_greedy_cost() / 2.0

def get_unfair_greedy_cost():
    global rank
    centers = []
    for i in range(rank):
        next_center = None
        dist = -1
        start_new_pass()
        point = get_next_point()
        if len(centers) == 0:
            centers.append(point)
            point = get_next_point()
        while point is not None:
            temp_dist, _ = get_distance_points(point, centers)
            if temp_dist > dist:
                next_center = point
                dist = temp_dist
            point = get_next_point()
        centers.append(next_center)
    return compute_cost(centers)


def get_opt_upper_bound():
    arb_solution = get_arb_solution()
    if len(arb_solution) == 0:
        return -1
    return compute_cost(arb_solution)


def extend_solution(solution, candidates):
    global capacities
    global rank
    residual_capacities = capacities.copy()
    remaining_capacity = rank
    for point in solution:
        residual_capacities[point.group] -= 1
        remaining_capacity -= 1
    for point in candidates:
        if residual_capacities[point.group] > 0 and point not in solution:
            solution.append(point)
            remaining_capacity -= 1
            residual_capacities[point.group] -= 1
        if remaining_capacity == 0:
            break
    return solution


def two_pass(eps):
    global rank
    lb = get_simple_opt_lower_bound()
    arb_solution = get_arb_solution()
    if len(arb_solution) == 0:
        return -1, [], -1
    ub = compute_cost(arb_solution)
    guesses = []
    lbp = lb
    while lbp <= ub * (1.0 + eps):
        guesses.append(lbp)
        lbp *= 1.0 + eps
    representatives_list = []
    pivots_list = []
    cost_list = []
    start_new_pass()
    point = get_next_point()
    for guess in guesses:
        representatives = {}
        pivots_list.append([point])
        representatives[point.id] = {}
        cost_list.append(float("inf"))
        representatives_list.append(representatives)
    if print_logs:
        print("Initialized the two-pass routine. #guesses=", len(guesses))
    point = get_next_point()
    while point is not None:
        # print("First pass index=", point.id)
        for index, (guess, pivots, representatives) in enumerate(zip(guesses, pivots_list, representatives_list)):
            if cost_list[index] < 0:
                continue
            dist = float("inf")
            for pivot in pivots:
                dist = min(dist, get_distance(point, pivot))
            if dist > 2 * guess:
                pivots.append(point)
                representatives[point.id] = {}
            if len(pivots) > rank:
                cost_list[index] = -1.0
        point = get_next_point()
    if print_logs:
        print("Starting second pass.")
    start_new_pass()
    point = get_next_point()
    while point is not None:
        # print("Second pass index=", point.id)
        for index, (guess, pivots, representatives) in enumerate(zip(guesses, pivots_list, representatives_list)):
            if cost_list[index] < 0:
                continue
            for pivot in pivots:
                if get_distance(point, pivot) <= guess:
                    representatives[pivot.id][point.group] = point
        point = get_next_point()
    for index, (guess, pivots, representatives) in enumerate(zip(guesses, pivots_list, representatives_list)):
        # print("After second pass index=", index)
        if cost_list[index] < 0:
            continue
        solution = compute_solution_for_pivots_and_reps(pivots, representatives)
        if len(solution) > 0:
            solution = extend_solution(solution,  arb_solution)
            return compute_cost(solution), solution, guess
    return -1, [], -1  # Will not come to this.


def compute_solution_for_pivots_and_reps(pivots, representatives):
    global g
    global capacities
    flow_graph = nx.DiGraph()
    for pivot in pivots:
        flow_graph.add_edge("s", "l" + str(pivot.id), capacity=1)
    for i in range(g):
        flow_graph.add_edge("r" + str(i), "t", capacity=capacities[i])
    for i in range(g):
        for pivot in pivots:
            if representatives[pivot.id].get(i) is not None:
                flow_graph.add_edge("l" + str(pivot.id), "r" + str(i), capacity=1)
    flow_value, flow_dict = nx.maximum_flow(flow_graph, "s", "t")
    if flow_value != len(pivots):
        return []
    solution = []
    for pivot in pivots:
        flow_edges_dict = flow_dict["l" + str(pivot.id)]
        for edge in flow_edges_dict.keys():
            if flow_edges_dict[edge] == 1:
                rep_group = int(edge[1:])
                rep = representatives[pivot.id][rep_group]
                solution.append(rep)
    caps = capacities.copy()
    # Sanity check
    for point in solution:
        caps[point.group] -= 1
        if caps[point.group] < 0:
            print(solution)
    return solution


# given a list of points, output a maximal sublist such that
# the pairwise distance of all points in the sublist is more than dist
def far_apart_points(points, dist):
    global rank
    rand_first_point = points[0]
    maximal_points = [rand_first_point]
    for point in points:
        temp_dist, temp_point = get_distance_points(point, maximal_points)
        if temp_dist > dist:
            maximal_points.append(point)
            if len(maximal_points) > rank:
                return []

    return maximal_points


# given a point q and a list of points, output ones that are
# within dist from q
def close_points(q, points, dist):
    closepoints = []
    for point in points:
        if get_distance(q, point) <= dist:
            closepoints.append(point)
    return closepoints


def distributed_referee_geom(kpivots, krepresentatives, tau, eps):
    if print_logs:
        print("#kpivots =", len(kpivots))
    pivots = []
    rep_points = []
    # loop to compute the union of the pivots and representatives
    for i in range(len(kpivots)):
        # print(i)
        # combining the i-th pivot set with the previous (i-1) pivot sets
        for kpiv in kpivots[i][:-1]:
            pivots.append(kpiv)
        # combining the i-th set of representatives
        # with all the previous (i-1) set of representatives
        # This part seems inefficient.  Why keep updating a dictionary and use it
        # if you can work on source dictionaries directly in this loop
        #representatives.update(krepresentatives[i])
        for reps in list(krepresentatives[i].values()):
            for rep in list(reps.values()):
                rep_points.append(rep)
    if print_logs:
        print("#reps =", len(rep_points))

    #print('done aggregating')
    arb_solution = get_arb_solution_from_points(rep_points)
    # geometric search procedure
    guess = tau / 5.1
    # find a maximal set of pivots that are 10*guess apart
    while True:
        # print("Processing guess #", ind)
        far_points = far_apart_points(pivots, 10 * guess)
        # find a set of representatives from each group
        # close to the subset of pivots picked earlier
        # print("far_apart_points(() done.")
        if len(far_points) == 0:
            guess *= 1.0 + eps
            continue
        # find a set of representatives from each group
        # close to the subset of pivots picked earlier
        close_representatives = {}
        for pt in far_points:
            close_representatives[pt.id] = {}
            closepoints = close_points(pt, rep_points, 5 * guess)
            for clpt in closepoints:
                close_representatives[pt.id][clpt.group] = clpt

        # run the max-flow procedure to obtain a solution
        # print(len(far_points),len(close_representatives))
        solution = compute_solution_for_pivots_and_reps(far_points, close_representatives)
        if len(solution) > 0:
            # print("Found solution.")
            solution = extend_solution(solution, arb_solution)
            return compute_cost(solution), solution, guess
        guess *= 1.0 + eps

def distributed_node(points):
    l = len(points)
    representatives = {}
    global rank
    rand_first_pivot = points[0]
    # initialize the first pivot to a uniformly random point
    pivots = [rand_first_pivot]
    representatives[rand_first_pivot.id] = {}
    # loop iterates r times
    # in each iteration, the point farthest away
    # from current set of pivots is added to the set of pivots
    for i in range(min(rank, l - 1)):
        dist = -1
        next_pivot = []
        # identifying the point farthest away from the current set of pivots
        for point in points:
            temp_dist, temp_point = get_distance_points(point, pivots)
            if temp_dist > dist:
                dist = temp_dist
                next_pivot = point
        pivots.append(next_pivot)
        # not creating a representative for the (r+1)-th pivot
        # the last pivot is only for calculating tau (for the node and the referee)
        if i != rank - 1:
            representatives[next_pivot.id] = {}

    # tau = half the distance of the last pivot point from the rest of the points
    tau, _ = get_distance_points(pivots[-1], pivots[:-1])
    tau = tau / 2

    # assigning the representatives for all but the (r+1)-th pivot
    for point in points:
        for pivot in pivots[:-1]:
            if get_distance(point, pivot) <= 2 * tau:
                representatives[pivot.id][point.group] = point

    return pivots[:-1], representatives, tau


def distributed_node_cores(player_points, q):
    pivots, representatives, tau = distributed_node(player_points)
    q.put((pivots, representatives, tau))


# Keep blocksize such that num_of_cores times blocksize data fits into memory
# IMPORTANT: blocksize x num_of_cores must always be equal to n.
def distributed_kcenter(blocksize, eps, num_of_cores=1):
    if print_logs:
        print("Number of cores using =", num_of_cores)
    kpivots = []
    krepresentatives = []
    start_new_pass()
    point = get_next_point()
    index = 1
    player_points = []
    q = Queue()
    p = []
    maxtau = 0
    num_unprocessed_blocks = 0
    while point is not None:
        # print("In distributed_kcenter().  Point index=", point.id)
        player_points.append(point)
        # i-th player gets a set of points indexed by {(blocks)*i,...,(blocks)(i+1)-1}
        # after applying the permutation
        if index % blocksize == 0:
            # call to the distributed_node to receive the message from the player
            proc = Process(target=distributed_node_cores, args=(player_points, q))
            p.append(proc)
            proc.start()
            player_points = []
            num_unprocessed_blocks += 1
        if num_unprocessed_blocks == num_of_cores:
            if print_logs:
                print("Waiting for", len(p), "blocks to finish.  Num = ", index / (blocksize * num_of_cores))
            for i in range(len(p)):
                # set block=True to block until we get a result
                result = q.get(True)
                kpivots.append(result[0])
                krepresentatives.append(result[1])
                maxtau = max(maxtau, result[2])
            q = Queue()
            p = []
            # gc.collect()
            num_unprocessed_blocks = 0
        index += 1
        point = get_next_point()
    if print_logs:
        print("Before call to distributed_referee_geom().")
    # call to the distributed_referee to compute the k-centers that constitute the output
    return distributed_referee_geom(kpivots, krepresentatives, maxtau, eps)

def get_all_points_distance_matrix():
    start_new_pass()
    point = get_next_point()
    points = []
    while point is not None:
        points.append(point)
        point = get_next_point()
    n = len(points)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i][j] = get_distance(points[i], points[j])
    return points, d


def get_all_points():
    start_new_pass()
    point = get_next_point()
    points = []
    while point is not None:
        points.append(point)
        point = get_next_point()
    return points


def kleindessner_etal():
    global capacities
    points, d = get_all_points_distance_matrix()
    groups = []
    n = len(points)
    for i in range(n):
        groups.append(points[i].group)
    kl_centers = kl.fair_k_center_APPROX(d, np.array(groups),
                                         np.array(capacities), np.array([],dtype=int))
    kl_solution = []
    for point in points:
        if point.id in kl_centers:
            kl_solution.append(point)
    kl_solution = extend_solution(kl_solution, get_arb_solution())
    return compute_cost(kl_solution), kl_solution


def chen_etal(guess, points):
    global rank
    global capacities
    arb_solution = get_arb_solution()
    partition = {}
    pivots = [points[0]]
    partition[points[0].id] = []
    for point in points:
        dist = float("inf")
        for pivot in pivots:
            dist = min(dist, get_distance(point, pivot))
        if dist > 2 * guess:
            pivots.append(point)
            partition[point.id] = []
        if len(pivots) > rank:
            return -1, []
    for point in points:
        for pivot in pivots:
            if get_distance(point, pivot) <= guess:
                partition[pivot.id].append(point)
    solution_chen_etal = partition_matroid_intersection(pivots, partition)
    if len(solution_chen_etal) == 0:
        return -1, []
    solution_chen_etal = extend_solution(solution_chen_etal, arb_solution)
    return compute_cost(solution_chen_etal), solution_chen_etal


def partition_matroid_intersection(pivots, partition):
    global g
    global capacities
    flow_graph = nx.DiGraph()
    for pivot in pivots:
        for rep in partition[pivot.id]:
            flow_graph.add_edge("s", "l" + str(pivot.id), capacity=1)
            flow_graph.add_edge("l" + str(pivot.id), "m" + str(rep.id), capacity=1)
            flow_graph.add_edge("m" + str(rep.id), "r" + str(rep.group), capacity=1)
    for i in range(g):
        flow_graph.add_edge("r" + str(i), "t", capacity=capacities[i])
    flow_value, flow_dict = nx.maximum_flow(flow_graph, "s", "t")
    if flow_value != len(pivots):
        return []
    solution_chen_etal = []
    for pivot in pivots:
        for rep in partition[pivot.id]:
            rep_dict = flow_dict["m" + str(rep.id)]
            for edge in rep_dict.keys():
                if rep_dict[edge] == 1:
                    solution_chen_etal.append(rep)
    caps = capacities.copy()
    for point in solution_chen_etal:
        caps[point.group] -= 1
        if caps[point.group] < 0:
            print(solution_chen_etal)
    return solution_chen_etal


def chen_etal_main_binary():
    global capacities
    start_new_pass()
    point = get_next_point()
    points = []
    while point is not None:
        points.append(point)
        point = get_next_point()
    n = len(points)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i][j] = get_distance(points[i], points[j])
    distances = d[np.triu_indices(n)]
    distances = np.sort(distances)
    distances = distances[n:]
    low = 0
    high = len(distances)
    solution_cost = float("inf")
    solution_centers = []
    while low <= high:
        ind = math.floor((low + high) / 2)
        guess = distances[ind]
        cost, centers = chen_etal(guess, points)
        if cost < 0:
            low = ind + 1
        else:
            if cost < solution_cost:
                solution_cost = cost
                solution_centers = centers
            high = ind - 1
    return solution_cost, solution_centers


def chen_etal_main_linear():
    points, d = get_all_points_distance_matrix()
    n = len(points)
    distances = d[np.triu_indices(n)]
    distances = np.sort(distances)
    distances = distances[n:]
    for guess in distances:
        cost, centers = chen_etal(guess, points)
        if cost >= 0:
            return cost, centers


def chen_etal_main_geometric(eps):
    points = get_all_points()
    guess = get_opt_lower_bound()
    while True:
        cost, centers = chen_etal(guess, points)
        if cost >= 0:
            return cost, centers
        guess *= 1.0 + eps


def kale(eps):
    global rank
    global capacities
    lb = get_simple_opt_lower_bound()
    arb_solution = get_arb_solution()
    if len(arb_solution) == 0:
        return -1, [], -1
    ub = compute_cost(arb_solution)
    guesses = []
    lbp = lb
    while lbp <= ub * (1.0 + eps):
        guesses.append(lbp)
        lbp *= 1.0 + eps
    representatives_list = []
    pivots_list = []
    cost_list = []
    residual_capacities_dict_list = []
    start_new_pass()
    point = get_next_point()
    for guess in guesses:
        representatives = {}
        pivots_list.append([point])
        representatives[point.id] = []
        cost_list.append(float("inf"))
        representatives_list.append(representatives)
        residual_capacities_dict = {}
        residual_capacities_dict[point.id] = capacities.copy()
        residual_capacities_dict_list.append(residual_capacities_dict)
    if print_logs:
        print("Initialized Kale's two-pass routine. #guesses=", len(guesses))
    point = get_next_point()
    while point is not None:
        # print("First pass index=", point.id)
        for index, (guess, pivots, representatives, residual_capacities_dict) in enumerate(zip(guesses, pivots_list, representatives_list, residual_capacities_dict_list)):
            if cost_list[index] < 0:
                continue
            dist = float("inf")
            for pivot in pivots:
                dist = min(dist, get_distance(point, pivot))
            if dist > 2 * guess:
                pivots.append(point)
                representatives[point.id] = []
                residual_capacities_dict[point.id] = capacities.copy()
            if len(pivots) > rank:
                cost_list[index] = -1.0
        point = get_next_point()
    if print_logs:
        print("Starting second pass of Kale.")
    start_new_pass()
    point = get_next_point()
    while point is not None:
        # print("Second pass index=", point.id)
        for index, (guess, pivots, representatives, residual_capacities_dict) in enumerate(zip(guesses, pivots_list, representatives_list, residual_capacities_dict_list)):
            if cost_list[index] < 0:
                continue
            for pivot in pivots:
                rem_caps = residual_capacities_dict[pivot.id]
                if rem_caps[point.group] > 0 and get_distance(point, pivot) <= guess:
                    representatives[pivot.id].append(point)
                    rem_caps[point.group] -= 1
        point = get_next_point()
    for index, (guess, pivots, representatives) in enumerate(zip(guesses, pivots_list, representatives_list)):
        # print("After second pass index=", index)
        if cost_list[index] < 0:
            continue
        solution = partition_matroid_intersection(pivots, representatives)
        if len(solution) > 0:
            solution = extend_solution(solution, arb_solution)
            return compute_cost(solution), solution, guess
    return -1, [], -1 # Will not come to this.

def sanity_check(solution):
    global capacities
    caps = capacities.copy()
    for point in solution:
        caps[point.group] -= 1
        if caps[point.group] < 0:
            print(solution)


# Uncomment the following for computing the time required to run on
# dataset = 'celeba'
# data_file_name = 'img_align_celeba_features.dat'
# group_file_name = 'list_attr_celeba.txt'
# g = 4
# capacities = [100] * g
# rank = sum(capacities)
# att1 = 21 # Male/Female
# att2 = 40 # Young/Not-young
# eps = 0.1
# blocksize = 10000
# print("Opt upper bound =", get_opt_upper_bound())
# start_two_pass = time.time()
# cost_two_pass, solution_two_pass, guess_two_pass = two_pass(eps)
# start_dist = time.time()
# cost_dist, solution_dist, guess_dist = distributed_kcenter(blocksize, eps)
# end_dist = time.time()
# print("Two pass: ", cost_two_pass, guess_two_pass, start_dist - start_two_pass)
# print("Distributed: ", cost_dist, guess_dist, end_dist - start_dist)

# for i in range(1, 41):
#     for j in range(i + 1, 41):
#         att1 = i
#         att2 = j
#         eps = 0.1
#         blocksize = 25
#         print(i, j)

def process_adult_dataset():
    f = open('uci-adult/adult.data', 'r')
    dataset = []
    real_indices = [0, 2, 4, 10, 11, 12]
    for line in f:
        coordinates = []
        fields = line.split(', ')
        if line == '\n':
            break
        for index in real_indices:
            coordinates.append(float(fields[index]))
        dataset.append(coordinates)
    f.close()
    dataset_np = np.array(dataset)
    normed = (dataset_np - dataset_np.mean(axis=0))/dataset_np.std(axis=0)
    np.savetxt("uci-adult/adult_normed.csv", normed, delimiter=",")

# process_adult_dataset()  # One time call.

def generate_random_euclidean_dataset():
    N = 40000
    f = open("/tmp/euclidean.csv", "ab")
    for i in range(100):
        print("Processed ", i)
        random_data = np.random.uniform(0, 100000, (N, 1000))
        np.savetxt(f, random_data, delimiter=",")
    f.close()
# generate_random_euclidean_dataset()

def generate_random_euclidean_dataset_groups():
    N = 4000000
    f = open("/tmp/euclidean_groups.txt", "w")
    for i in range(4000000):
        group = np.random.randint(4)
        f.write(str(group) + "\n")
    f.close()
# generate_random_euclidean_dataset_groups()

def time_to_read_data_file():
    start = time.time()
    global data_file_name
    f = open(data_file_name, "r")
    for line in f:
        coordinates = line.split(',')
    f.close()
    return time.time() - start

def run_algorithms():
    global dataset
    global capacities
    global eps
    global blocksize
    arb_solution = get_arb_solution()
    if len(arb_solution) == 0:
        print(dataset, "&", capacities, "&", "No feasible solution satisfying capacity constraint exists.  Not running for this setting")
        return
    #print("Opt upper bound =", get_opt_upper_bound())
    #print("Unfair greedy cost =", get_unfair_greedy_cost())
    lb = get_opt_lower_bound()

    start_kale = time.time()
    cost_kale, solution_kale, guess_kale = kale(eps)
    start_two_pass = time.time()
    #print("Kale: ", cost_kale, guess_kale, start_two_pass - start_kale)
    cost_two_pass, solution_two_pass, guess_two_pass = two_pass(eps)
    start_dist = time.time()
    #print("Two pass: ", cost_two_pass, guess_two_pass, start_dist - start_two_pass)
    cost_dist, solution_dist, guess_dist = distributed_kcenter(blocksize, eps)
    end_dist = time.time()
    #print("Distributed: ", cost_dist, guess_dist, end_dist - start_dist)

    # Run only for small number of points
    cost_kl, solution_kl = kleindessner_etal()
    #print("Kleindessner et al: ", cost_kl, solution_kl)

    # Run only for small number of points
    # cost_chen_etal, solution_chen_etal = chen_etal_main_binary()
    #print("Chen et al, binary search: ", cost_chen_etal, solution_chen_etal)

    # Run only for small number of points
    # cost_chen_etal_lin, solution_chen_etal_lin = chen_etal_main_linear()
    # print("Chen et al, linear search: ", cost_chen_etal_lin, solution_chen_etal_lin)

    # Run only for small number of points
    cost_chen_etal_geom, solution_chen_etal_geom = chen_etal_main_geometric(eps)
    # print("Chen et al, geometric search: ", cost_chen_etal_geom, solution_chen_etal_geom)

    sanity_check(solution_kale)
    sanity_check(solution_two_pass)
    sanity_check(solution_dist)
    sanity_check(solution_kl)
    sanity_check(solution_chen_etal_geom)
    # sanity_check(solution_chen_etal_lin)
    print(dataset, "&", capacities, "&", round(lb, 2), "&", round(cost_chen_etal_geom / lb, 2), "&", round(cost_kale / lb, 2),
          "&", round(cost_kl / lb, 2), "&", round(cost_two_pass / lb, 2), "&", round(cost_dist / lb, 2), "\\\\ \\hline")


print("Dataset & Capacities & Lower Bound & Chen et al. & Kale & Kleindessner et al & Two pass & Distributed \\\\")
att1 = 21
eps = 0.1
blocksize = 25
num_recs_to_read = 1000

datasets = ['celeba', 'sushia', 'sushib', 'adult']
data_file_names = ['img_align_celeba_features.dat', 'sushi3-2016/sushi3a.5000.10.order', 'sushi3-2016/sushi3b.5000.10.score', 'uci-adult/adult_normed.csv']
group_file_names = ['list_attr_celeba.txt', 'sushi3-2016/sushi3.udata', 'sushi3-2016/sushi3.udata', 'uci-adult/adult_attr.data']
for dataset, data_file_name, group_file_name in zip(datasets, data_file_names, group_file_names):
    g = 2
    capacities = [2] * g
    rank = sum(capacities)
    # print(dataset, "Attribute Male/Female, Capacities = ", capacities)
    run_algorithms()

dataset = 'celeba'
data_file_name = 'img_align_celeba_features.dat'
group_file_name = 'list_attr_celeba.txt'
num_recs_to_read = 1000
att2 = 40
g = 4
capacities = [2] * g
rank = sum(capacities)
# print(dataset, "Attribute Male/Female & Young/Not-young, Capacities = ", capacities)
run_algorithms()

datasets = ['sushia', 'sushib']
data_file_names = ['sushi3-2016/sushi3a.5000.10.order', 'sushi3-2016/sushi3b.5000.10.score']
group_file_names = ['sushi3-2016/sushi3.udata', 'sushi3-2016/sushi3.udata']
for dataset, data_file_name, group_file_name in zip(datasets, data_file_names, group_file_names):
    for g in range(6, 13, 6):
        capacities = [2] * g
        rank = sum(capacities)
        # print(dataset, "Attribute Male/Female & Age_Group, Capacities = ", capacities)
        run_algorithms()

dataset = 'adult'
data_file_name = 'uci-adult/adult_normed.csv'
group_file_name = 'uci-adult/adult_attr.data'
for g in range(5, 11, 5):
    capacities = [2] * g
    rank = sum(capacities)
    # print(dataset, "Attribute Male/Female & Race, Capacities = ", capacities)
    run_algorithms()


# Uncomment the following code for running on the 100 GB dataset

# print_logs = True
# dataset = 'random_euclidean'
# data_file_name = '/tmp/euclidean.csv'
# group_file_name = '/tmp/euclidean_groups.txt'
# g = 4
# capacities = [2] * g
# rank = sum(capacities)
# #print("Opt lower bound =", get_opt_lower_bound(), "Opt upper bound =", get_opt_upper_bound())
# eps = 0.1
# blocksize = 10000
#
# # num_recs_to_read = 4000
# #print("Time to read data file =", time_to_read_data_file())
#
# start_dist = time.time()
# cost_dist, solution_dist, guess_dist = distributed_kcenter(blocksize, eps, 4)
# end_dist = time.time()
# print("Distributed: ", cost_dist, guess_dist, end_dist - start_dist)
# start_two_pass = time.time()
# cost_two_pass, solution_two_pass, guess_two_pass = two_pass(eps)
# end_two_pass = time.time()
# print("Two pass: ", cost_two_pass, guess_two_pass, end_two_pass - start_two_pass)
#
# start_kale = time.time()
# cost_kale, solution_kale, guess_kale = kale(eps)
# end_kale = time.time()
# print("Kale: ", cost_kale, guess_kale, end_kale - start_kale)

# Uncomment the following code to get results in Table 2
#
# print("Dataset & Capacities & Lower Bound & Chen et al. & Kale & Kleindessner et al & Two pass & Distributed \\\\")
# att1 = 21
# eps = 0.1
# blocksize = 25
# num_recs_to_read = 1000
#
# datasets = ['celeba', 'sushia', 'sushib', 'adult']
# data_file_names = ['img_align_celeba_features.dat', 'sushi3-2016/sushi3a.5000.10.order', 'sushi3-2016/sushi3b.5000.10.score', 'uci-adult/adult_normed.csv']
# group_file_names = ['list_attr_celeba.txt', 'sushi3-2016/sushi3.udata', 'sushi3-2016/sushi3.udata', 'uci-adult/adult_attr.data']
# for dataset, data_file_name, group_file_name in zip(datasets, data_file_names, group_file_names):
#     g = 2
#     capacities = [1, 3]
#     rank = sum(capacities)
#     #print(dataset, "Attribute Male/Female, Capacities = ", capacities)
#     run_algorithms()
#     capacities = [3, 1]
#     rank = sum(capacities)
#     #print(dataset, "Attribute Male/Female, Capacities = ", capacities)
#     run_algorithms()
#
# dataset = 'celeba'
# data_file_name = 'img_align_celeba_features.dat'
# group_file_name = 'list_attr_celeba.txt'
# num_recs_to_read = 1000
# att2 = 40
# g = 4
# capacities = [1, 1, 3, 3]
# rank = sum(capacities)
# #print(dataset, "Attribute Male/Female & Young/Not-young, Capacities = ", capacities)
# run_algorithms()
#
# capacities = [3, 3, 1, 1]
# rank = sum(capacities)
# #print(dataset, "Attribute Male/Female & Young/Not-young, Capacities = ", capacities)
# run_algorithms()
#
#
# datasets = ['sushia', 'sushib']
# data_file_names = ['sushi3-2016/sushi3a.5000.10.order', 'sushi3-2016/sushi3b.5000.10.score']
# group_file_names = ['sushi3-2016/sushi3.udata', 'sushi3-2016/sushi3.udata']
# for dataset, data_file_name, group_file_name in zip(datasets, data_file_names, group_file_names):
#     g = 6
#     capacities = [1, 1, 1, 3, 3, 3]
#     rank = sum(capacities)
#     # print(dataset, "Attribute Male/Female & Age_Group, Capacities = ", capacities)
#     run_algorithms()
#     capacities = [3, 3, 3, 1, 1, 1]
#     rank = sum(capacities)
#     # print(dataset, "Attribute Male/Female & Age_Group, Capacities = ", capacities)
#     run_algorithms()
#     g = 12
#     capacities = [1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 3, 3]
#     rank = sum(capacities)
#     # print(dataset, "Attribute Male/Female & Age_Group, Capacities = ", capacities)
#     run_algorithms()
#
#     g = 12
#     capacities = [3, 3, 3, 1, 1, 1, 3, 3, 3, 1, 1, 1]
#     rank = sum(capacities)
#     # print(dataset, "Attribute Male/Female & Age_Group, Capacities = ", capacities)
#     run_algorithms()
#
# dataset = 'adult'
# data_file_name = 'uci-adult/adult_normed.csv'
# group_file_name = 'uci-adult/adult_attr.data'
# g = 5
# capacities = [1, 1, 3, 3, 3]
# rank = sum(capacities)
# # print(dataset, "Attribute Male/Female & Race, Capacities = ", capacities)
# run_algorithms()
# capacities = [3, 3, 1, 1, 1]
# rank = sum(capacities)
# # print(dataset, "Attribute Male/Female & Race, Capacities = ", capacities)
# run_algorithms()
#
# g = 10
# capacities = [1, 1, 3, 3, 3, 1, 1, 3, 3, 3]
# rank = sum(capacities)
# # print(dataset, "Attribute Male/Female & Race, Capacities = ", capacities)
# run_algorithms()
# capacities = [1, 1, 3, 3, 3, 3, 3, 1, 1, 1]
# rank = sum(capacities)
# # print(dataset, "Attribute Male/Female & Race, Capacities = ", capacities)
# run_algorithms()
#
