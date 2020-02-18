import math
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 20,
          'legend.handlelength': 7}
plt.rcParams.update(params)

from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

import kleindessner_etal_algorithms as kl

from multiprocessing import Process, Queue


def random_graph_erdos_renyi(n, p, w):
    """
    function to output the adjacency matrix of a n-vertex graph
    in which each edge is sample independently with probability p
    and is given weight uniformly in (0, w).
    """
    adj = np.zeros((n, n))  # first create an n by n matrix with 0 entries
    for i in range(n):
        for j in range(i):
            # sampling a uniformly random number between 0 and 1
            r = np.random.uniform()
            # ensuring that the probability of an edge is p
            if r <= p:
                random_weight = np.random.uniform(0.0, w)
                adj[i][j] = random_weight  # adding the edge
                adj[j][i] = random_weight  # adding the edge
    return adj


def random_metric_erdos_renyi(n, p, w):
    """
    function to output a random distance matrix based on Erdos-Renyi
    graph construction on n vertices with edge-existence probability p,
    and random edge weight uniformly in (0,w).
    """
    # Construct a random graph first.
    adj = random_graph_erdos_renyi(n, p, w)

    # Construct a networkx graph using the above.
    graph = nx.Graph()
    nodes = [i for i in range(n)]
    graph.add_nodes_from(nodes)
    for i in range(n):
        for j in range(n):
            edge_weight = adj[i][j]
            if edge_weight > 0:
                graph.add_edge(i, j, weight=edge_weight)

    # Compute the shortest-path metric using Bellman-Ford algorithm in networkx.
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph))

    # Populate d.
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = lengths[i].get(j)
            if dist is None:
                d[i][j] = float("inf")
            else:
                d[i][j] = dist
    return d


class RandomFairKCenter:
    """
    Given parameters n, p, w, probabilities with which each group should be assigned,
    and partition-matroid capacities (one for each group), construct a random instance.
    """

    def __init__(self, n, probabilities, capacities, metric_generation):

        self.n = n

        # Assign the number of groups.
        self.g = len(probabilities)

        # An n length list of groups denoting who gets which of the g groups.
        self.groups = [-1] * n

        # Capacity for each group.
        self.capacities = [0] * self.g

        # Assign groups randomly based on the given probabilities independently
        # for each of the n points.
        cumulative_probabilities = [None] * self.g
        cumulative_probability = 0.0
        for i in range(self.g):
            cumulative_probabilities[i] = cumulative_probability + probabilities[i]
            cumulative_probability = cumulative_probabilities[i]
        for i in range(n):
            random_number = np.random.uniform(0, 1)
            for j in range(self.g):
                if random_number <= cumulative_probabilities[j]:
                    self.groups[i] = j
                    self.capacities[j] += 1
                    break

        # Doing this to be safe.  Just in case some of the randomly assigned groups
        # is assigned less number of times than allowed capacity.
        for i in range(self.g):
            self.capacities[i] = min(self.capacities[i], capacities[i])

        self.rank = sum(self.capacities)

        # Compute random metric d.
        self.metric = metric_generation[0]
        p = metric_generation[1]
        w = metric_generation[2]
        self.d = random_metric_erdos_renyi(n, p, w)

    def get_distance(self, i, j):
        return self.d[i][j]

    def compute_cost(self, solution):
        cost = 0.0
        for i in range(self.n):
            dist = float("inf")
            for point in solution:
                dist = min(dist, self.get_distance(i, point))
            cost = max(cost, dist)
        return cost

    def extend_solution(self, solution, candidates):
        residual_capacities = self.capacities.copy()
        remaining_capacity = self.rank
        for point in solution:
            residual_capacities[self.groups[point]] -= 1
            remaining_capacity -= 1
        for point in candidates:
            group = self.groups[point]
            if residual_capacities[group] > 0 and point not in solution:
                solution.append(point)
                remaining_capacity -= 1
                residual_capacities[group] -= 1
            if remaining_capacity == 0:
                break
        return solution

    def get_arb_solution(self, points):
        arb_solution = []
        residual_capacities = self.capacities.copy()
        total_residual_capacity = sum(residual_capacities)
        for point in points:
            if residual_capacities[self.groups[point]] > 0:
                arb_solution.append(point)
                residual_capacities[self.groups[point]] -= 1
                total_residual_capacity -= 1
                if total_residual_capacity == 0:
                    return arb_solution
        return []

    def two_pass(self, guess):
        representatives = {}
        pivots = [0]
        representatives[0] = {}
        arb_solution = []
        residual_capacities = self.capacities.copy()
        for i in range(self.n):
            dist = float("inf")
            for pivot in pivots:
                dist = min(dist, self.get_distance(i, pivot))
            if dist > 2 * guess:
                pivots.append(i)
                representatives[i] = {}
            if len(pivots) > self.rank:
                return []
        for i in range(self.n):
            for pivot in pivots:
                if self.get_distance(i, pivot) <= guess:
                    representatives[pivot][self.groups[i]] = i
            if residual_capacities[self.groups[i]] > 0:
                arb_solution.append(i)
                residual_capacities[self.groups[i]] -= 1
        solution = self.compute_solution_for_pivots_and_reps(pivots, representatives)
        if len(solution) == 0:
            return []
        return self.extend_solution(solution, arb_solution)

    def two_pass_main_log(self, eps):

        # A lower bound on optimum using using Gonzalez's farthest point heuristic.

        lb = self.get_simple_opt_lower_bound()
        arb_solution = self.get_arb_solution(range(self.n))
        ub = self.compute_cost(arb_solution)
        prev_solution = []
        while ub/lb >= 1 + eps:
            guess = (ub * lb) ** 0.5
            solution = self.two_pass(guess)
            if len(solution) > 0:
                prev_solution = solution
                ub = guess
            else:
                lb = guess
        solution = self.extend_solution(prev_solution, arb_solution)
        return self.compute_cost(solution), solution

    def two_pass_main2(self, eps):
        arb_solution = self.get_arb_solution(range(self.n))
        guess = self.compute_cost(arb_solution)
        prev_solution = []
        while True:
            solution = self.two_pass(guess)
            if len(solution) == 0:
                return self.compute_cost(prev_solution), prev_solution
            else:
                prev_solution = solution
            guess /= 1.0 + eps

    def compute_solution_for_pivots_and_reps(self, pivots, representatives):
        flow_graph = nx.DiGraph()
        for pivot in pivots:
            flow_graph.add_edge("s", "l" + str(pivot), capacity=1)
        for i in range(self.g):
            flow_graph.add_edge("r" + str(i), "t", capacity=self.capacities[i])
        for i in range(self.g):
            for pivot in pivots:
                if representatives[pivot].get(i) is not None:
                    flow_graph.add_edge("l" + str(pivot), "r" + str(i), capacity=1)
        start = time.time()
        flow_value, flow_dict = nx.maximum_flow(flow_graph, "s", "t")
        end = time.time()
        if flow_value != len(pivots):
            return []
        solution = []
        for pivot in pivots:
            flow_edges_dict = flow_dict["l" + str(pivot)]
            for edge in flow_edges_dict.keys():
                if flow_edges_dict[edge] == 1:
                    rep_group = int(edge[1:])
                    rep = representatives[pivot][rep_group]
                    solution.append(rep)
        caps = self.capacities.copy()
        for point in solution:
            caps[self.groups[point]] -= 1
            if caps[self.groups[point]] < 0:
                print(solution)
        return solution

    def get_simple_opt_lower_bound(self):
        points = range(self.rank + 1)
        min_dist = float("inf")
        for i in points:
            dist_i = float("inf")
            for j in points:
                cur_dist = self.get_distance(i, j)
                if j != i and cur_dist < dist_i:
                    dist_i = cur_dist
            if dist_i < min_dist:
                min_dist = dist_i
        return min_dist / 2.0

    def two_pass_main(self, eps):
        guess = self.get_simple_opt_lower_bound()

        while True:
            solution = self.two_pass(guess)
            if len(solution) > 0:
                return self.compute_cost(solution), solution
            guess *= 1.0 + eps



    ####################################################################################
    # distributed kcenter

    def get_distance_points(self, i, points):
        dist = float("inf")
        cert = -1
        for point in points:
            cur_dist = self.get_distance(i, point)
            if cur_dist < dist:
                dist = cur_dist
                cert = point
        return dist, cert

    def distributed_node(self, points):
        l = len(points)
        representatives = {}
        first_pivot = points[0]
        # initialize the first pivot to a uniformly random point
        pivots = [first_pivot]
        representatives[first_pivot] = {}
        # loop iterates r times
        # in each iteration, the point farthest away
        # from current set of pivots is added to the set of pivots
        for i in range(min(self.rank, l - 1)):
            dist = -1
            next_pivot = []
            # identifying the point farthest away from the current set of pivots
            for point in points:
                temp_dist, temp_point = self.get_distance_points(point, pivots)
                if temp_dist > dist:
                    dist = temp_dist
                    next_pivot = point
            pivots.append(next_pivot)
            # not creating a representative for the (r+1)-th pivot
            # the last pivot is only for calculating tau (for the node and the referee)
            if i != self.rank - 1:
                representatives[next_pivot] = {}

        # tau = half the distance of the last pivot point from the rest of the points
        tau, _ = self.get_distance_points(pivots[-1], pivots[:-1])
        tau = tau / 2

        # assigning the representatives for all but the (r+1)-th pivot
        for point in points:
            for pivot in pivots[:-1]:
                if self.get_distance(point, pivot) <= 2 * tau:
                    representatives[pivot][self.groups[point]] = point

        return pivots[:-1], representatives, tau

    # given a list of points, output a maximal sublist such that
    # the pairwise distance of all points in the sublist is more than dist
    def far_apart_points(self, points, dist):
        rand_first_point = points[0]
        maximal_points = [rand_first_point]
        for point in points:
            temp_dist, temp_point = self.get_distance_points(point, maximal_points)
            if temp_dist > dist:
                maximal_points.append(point)
                if len(maximal_points) > self.rank:
                    return []

        return maximal_points

    # given a point q and a list of points, output ones that are
    # within dist from q
    def close_points(self, q, points, dist):
        closepoints = []
        for point in points:
            if self.get_distance(q, point) <= dist:
                closepoints.append(point)
        return closepoints

    # Using rank many points that are "far" away using the greedy procedure
    def get_opt_lower_bound(self):
        # initialize the first pivot to be the first point
        centers = [0]
        for i in range(self.rank - 1):
            next_center = -1
            dist = -1
            for point in range(self.n):
                temp_dist, _ = self.get_distance_points(point, centers)
                if (temp_dist > dist):
                    next_center = point
                    dist = temp_dist

            centers.append(next_center)

        return self.compute_cost(centers) / 2.0

    # the referee as input gets a list of k pivots sets (lists)
    # and k sets (lists) of representatives
    def distributed_referee_geom(self, kpivots, krepresentatives, tau, eps):
        pivots = []
        rep_points = []
        # loop to compute the union of the pivots and representatives
        for i in range(len(kpivots)):
            # combining the i-th pivot set with the previous (i-1) pivot sets
            for kpiv in kpivots[i]:
                pivots.append(kpiv)
            # combining the i-th set of representatives
            # with all the previous (i-1) set of representatives

            # This part seems inefficient.  Why keep updating a dictionary and use it
            # if you can work on source dictionaries directly in this loop
            #representatives.update(krepresentatives[i])

            for reps in list(krepresentatives[i].values()):
                for rep in list(reps.values()):
                    rep_points.append(rep)
                # rep_points = rep_points + list(reps.values())
        arb_solution = self.get_arb_solution(rep_points)
        # geometric search procedure
        guess = tau / 5.1
        while True:
            # find a maximal set of pivots that are 10*guess apart
            far_points = self.far_apart_points(pivots, 10 * guess)
            if len(far_points) == 0:
                guess *= 1.0 + eps
                continue
            # find a set of representatives from each group
            # close to the subset of pivots picked earlier
            close_representatives = {}
            for pt in far_points:
                close_representatives[pt] = {}
                closepoints = self.close_points(pt, rep_points, 5 * guess)
                for clpt in closepoints:
                    close_representatives[pt][self.groups[clpt]] = clpt

            # run the max-flow procedure to obtain a solution
            solution = self.compute_solution_for_pivots_and_reps(far_points, close_representatives)
            if len(solution) > 0:
                solution = self.extend_solution(solution, arb_solution)
                return self.compute_cost(solution), solution
            guess *= 1.0 + eps

    def distributed_kcenter(self, points, m, eps):
        l = len(points)
        # first identify the number of partitions of the input
        blocks = math.ceil(l / m)
        kpivots = []
        krepresentatives = []
        maxtau = 0
        player_time = -1
        rf_time = 0
        dist_count = 0
        for i in range(m):
            # i-th player gets a set of points indexed by {(blocks)*i,...,(blocks)(i+1)-1}
            # after applying the permutation
            player_points = [points[j] for j in range(blocks * i, min(l, blocks * (i + 1)))]
            # call to the distributed_node to receive the message from the player
            startpl = time.time()
            pivots, representatives, tau = self.distributed_node(player_points)
            endpl = time.time()
            player_time = max(player_time, endpl - startpl)
            # aggregating all the messages received so far
            # time charged to the referee
            startrf = time.time()
            kpivots.append(pivots)
            krepresentatives.append(representatives)
            maxtau = max(maxtau, tau)
            endrf = time.time()
            rf_time += endrf - startrf
            dist_count += blocks
            if (dist_count >= l):
                break
        # call to the distributed_referee to compute the k-centers that constitute the output
        startrf = time.time()
        cost, solution = self.distributed_referee_geom(kpivots, krepresentatives, maxtau, eps)
        endrf = time.time()
        return cost, solution, (endrf - startrf + player_time + rf_time)

    ####################################################################################

    def chen_etal(self, guess):
        partition = {}
        rand_first_pivot = 0
        pivots = [rand_first_pivot]
        partition[rand_first_pivot] = []
        for i in range(self.n):
            dist = float("inf")
            for pivot in pivots:
                dist = min(dist, self.get_distance(i, pivot))
            if dist > 2 * guess:
                pivots.append(i)
                partition[i] = []
            if len(pivots) > self.rank:
                return -1, []
        for i in range(self.n):
            for pivot in pivots:
                if self.get_distance(i, pivot) <= guess:
                    partition[pivot].append(i)
        solution_chen_etal = self.partition_matroid_intersection(pivots, partition)
        if len(solution_chen_etal) == 0:
            return -1, []
        return self.compute_cost(solution_chen_etal), solution_chen_etal

    def partition_matroid_intersection(self, pivots, partition):
        flow_graph = nx.DiGraph()
        for pivot in pivots:
            for rep in partition[pivot]:
                flow_graph.add_edge("s", "l" + str(pivot), capacity=1)
                flow_graph.add_edge("l" + str(pivot), "m" + str(rep), capacity=1)
                flow_graph.add_edge("m" + str(rep), "r" + str(self.groups[rep]), capacity=1)
        for i in range(self.g):
            flow_graph.add_edge("r" + str(i), "t", capacity=self.capacities[i])
        flow_value, flow_dict = nx.maximum_flow(flow_graph, "s", "t")
        if flow_value != len(pivots):
            return []
        solution_chen_etal = []
        for pivot in pivots:
            for rep in partition[pivot]:
                rep_dict = flow_dict["m" + str(rep)]
                for edge in rep_dict.keys():
                    if rep_dict[edge] == 1:
                        solution_chen_etal.append(rep)
        caps = self.capacities.copy()
        for point in solution_chen_etal:
            caps[self.groups[point]] -= 1
            if caps[self.groups[point]] < 0:
                print(solution_chen_etal)
        return solution_chen_etal

    def chen_etal_main_binary(self):
        distances = self.d[np.triu_indices(self.n)]
        distances = np.sort(distances)
        distances = distances[self.n:]
        low = 0
        high = len(distances)
        solution_cost = float("inf")
        solution_centers = []
        while low <= high:
            ind = math.floor((low + high) / 2)
            guess = distances[ind]
            cost, centers = self.chen_etal(guess)
            if cost < 0:
                low = ind + 1
            else:
                if cost < solution_cost:
                    solution_cost = cost
                    solution_centers = centers
                high = ind - 1
        return solution_cost, solution_centers


    def chen_etal_main_geometric(self, eps):
        guess = self.get_simple_opt_lower_bound()

        while True:
            cost, centers = self.chen_etal(guess)
            if cost > 0:
                return cost, centers
            guess *= 1.0 + eps

    def distributed_node_cores(self, player_points, q):
        pivots, representatives, tau = self.distributed_node(player_points)
        q.put((pivots, representatives, tau))

    def distributed_kcenter_cores(self, points, m, eps):
        l = len(points)
        # first identify the number of partitions of the input
        blocks = math.ceil(l / m)
        kpivots = []
        krepresentatives = []
        q = Queue()
        p = []
        for i in range(m):
            # i-th player gets a set of points indexed by {(blocks)*i,...,(blocks)(i+1)-1}
            player_points = [points[j] for j in range(blocks * i, min(l, blocks * (i + 1)))]
            # call to the distributed_node to receive the message from the player
            p.append(Process(target=self.distributed_node_cores, args=(player_points, q)))
            p[i].start()
        # aggregating all the messages received so far
        maxtau = 0
        for i in range(m):
            # set block=True to block until we get a result
            result = q.get(True)
            kpivots.append(result[0])
            krepresentatives.append(result[1])
            maxtau = max(maxtau, result[2])
        # call to the distributed_referee to compute the k-centers that constitute the output
        return self.distributed_referee_geom(kpivots, krepresentatives, maxtau, eps)


    def kale(self, guess):
        partition = {}
        rand_first_pivot = 0
        pivots = [rand_first_pivot]
        partition[rand_first_pivot] = []
        residual_capacities_dict = {}
        residual_capacities_dict[rand_first_pivot] = self.capacities.copy()
        arb_solution = []
        residual_capacities = self.capacities.copy()
        for i in range(self.n):
            dist = float("inf")
            for pivot in pivots:
                dist = min(dist, self.get_distance(i, pivot))
            if dist > 2 * guess:
                pivots.append(i)
                partition[i] = []
                residual_capacities_dict[i] = self.capacities.copy()
            if len(pivots) > self.rank:
                return []
            if residual_capacities[self.groups[i]] > 0:
                arb_solution.append(i)
                residual_capacities[self.groups[i]] -= 1
        for i in range(self.n):
            for pivot in pivots:
                rem_caps = residual_capacities_dict[pivot]
                if rem_caps[self.groups[i]] > 0 and self.get_distance(i, pivot) <= guess:
                    partition[pivot].append(i)
                    rem_caps[self.groups[i]] -= 1
        solution_kale = self.partition_matroid_intersection(pivots, partition)
        if len(solution_kale) == 0:
            return []
        return self.extend_solution(solution_kale, arb_solution)

    def kale_main(self, eps):
        guess = self.get_simple_opt_lower_bound()

        while True:
            solution = self.kale(guess)
            if len(solution) > 0:
                return self.compute_cost(solution), solution
            guess *= 1.0 + eps

def two_pass(points_instance, eps):
    start = time.time()
    two_pass_cost, two_pass_centers = points_instance.two_pass_main(eps)
    end = time.time()
    return end - start, two_pass_cost


def kl_algorithm(points_instance):
    start = time.time()
    kl_centers = kl.fair_k_center_APPROX(points_instance.d, np.array(points_instance.groups),
                                         np.array(points_instance.capacities), np.array([],dtype=int))
    kl_cost = points_instance.compute_cost(kl_centers)
    end = time.time()
    return end - start, kl_cost


def distributed(points_instance, eps):
    start = time.time()
    dist_cost, _, dist_time = points_instance.distributed_kcenter(list(range(points_instance.n)), 4, eps)
    end = time.time()
    return end - start, dist_cost


def distributed_parallel(points_instance, eps):
    start = time.time()
    dist_p_cost, _ = points_instance.distributed_kcenter_cores(list(range(points_instance.n)), 4, eps)
    end = time.time()
    return end - start, dist_p_cost


def chen_etal(points_instance):
    start = time.time()
    chen_etal_cost, chen_etal_centers = points_instance.chen_etal_main_binary()
    end = time.time()
    return end - start, chen_etal_cost

def kale_expt(points_instance,eps):
    start = time.time()
    kale_cost, kale_centers = points_instance.kale_main(eps)
    end = time.time()
    return end - start, kale_cost


###############################################
# Experiments
###############################################
def instancesize_runtime_plot(nlist, reps, capacities):
    w = 1000.0
    eps = 0.1
    two_pass_time = []
    kl_time = []
    distributed_time = []
    chen_etal_time = []
    kale_time = []
    for n in nlist:
        two_pass_time_avg = 0
        kl_time_avg = 0
        dist_time_avg = 0
        chen_time_avg = 0
        kale_time_avg = 0
        for i in range(reps):
            print("n = ", n, "Rep #", i)
            metric_generation = "er", 2.0 * np.log(n) / n, w
            random_instance = RandomFairKCenter(n, [1 / len(capacities)] * len(capacities), capacities, metric_generation)
            t, _ = chen_etal(random_instance)
            chen_time_avg += t
            t, _ = kale_expt(random_instance, eps)
            kale_time_avg += t
            t, _ = two_pass(random_instance, eps)
            two_pass_time_avg += t
            t, _ = kl_algorithm(random_instance)
            kl_time_avg += t
            t, _ = distributed(random_instance, eps)
            dist_time_avg += t

        two_pass_time.append(two_pass_time_avg / reps)
        kl_time.append(kl_time_avg / reps)
        distributed_time.append(dist_time_avg / reps)
        kale_time.append(kale_time_avg / reps)
        chen_etal_time.append(chen_time_avg / reps)

    df1 = pd.DataFrame({"nlist":nlist,"Two Pass":two_pass_time,"Distributed":distributed_time,
                        "Kleindessner et al.":kl_time,"Kale":kale_time,"Chen et al.":chen_etal_time})
    df2 = pd.DataFrame({"nlist": nlist, "Two Pass": two_pass_time, "Distributed": distributed_time,
                        "Kleindessner et al.": kl_time, "Kale": kale_time})

    sns.set()
    #fig, axarr = plt.subplots(1,2)

    sns.lineplot(x="nlist",y="value", hue="Algorithms", style="Algorithms",
                 data=pd.melt(df1,['nlist'],var_name='Algorithms'),
                 markers=['o', 'H', 'v', 'P', '^'],dashes=[(1,1),(3,1),(2,2),(5,2),(6,3)],legend="full", markersize=15)
    plt.xlabel('Instance Size'+ '\n' + 'Capacities: '+ "["+str(capacities[0])+"] * "+str(len(capacities)) + '; #repetitions = ' + str(reps))
    plt.ylabel('Runtime (in seconds)')
    #plt.text(1,0,'capacities: '+",".join(str(c) for c in capacities))
    plt.tight_layout()
    plt.legend(markerscale=2)
    plt.show()
    plt.clf()
    sns.set()
    sns.lineplot(x="nlist", y="value", hue="Algorithms", style="Algorithms",
                 data=pd.melt(df2, ['nlist'], var_name='Algorithms'),
                 markers=['o', 'H', 'v', 'P', '^'], dashes=[(1,1),(3,1),(2,2),(5,2)], legend="full", markersize=15)
    plt.xlabel('Instance Size' + '\n' + 'Capacities: '+ "["+str(capacities[0])+"] * "+str(len(capacities)) + '; #repetitions = ' + str(reps))
    plt.ylabel('Runtime (in seconds)')
    #plt.text(1,0,'capacities: '+",".join(str(c) for c in capacities))
    plt.tight_layout()
    plt.legend(markerscale=2)
    plt.show()
    plt.clf()
    return


def approxratio_comparison_plot(n, capacities_list, reps):
    w = 1000.0
    eps = 0.1
    two_pass_cost = []
    kl_cost = []
    distributed_cost = []
    kale_cost = []
    chen_etal_cost = []
    for capacities in capacities_list:
        two_pass_cost_avg = 0
        kl_cost_avg = 0
        dist_cost_avg = 0
        kale_cost_avg = 0
        chen_cost_avg = 0
        lowerbound_cost_avg = 0
        for i in range(reps):
            print(capacities, "Rep #", i)
            metric_generation = "er", 2.0 * np.log(n) / n, w
            random_instance = RandomFairKCenter(n, [1 / len(capacities)] * len(capacities), capacities,
                                                metric_generation)
            _, c = chen_etal(random_instance)
            chen_cost_avg += c
            _, c = kale_expt(random_instance, eps)
            kale_cost_avg += c
            _, c = two_pass(random_instance, eps)
            two_pass_cost_avg += c
            _, c = kl_algorithm(random_instance)
            kl_cost_avg += c
            _, c = distributed(random_instance, eps)
            dist_cost_avg += c
            c = random_instance.get_opt_lower_bound()
            lowerbound_cost_avg += c

        two_pass_cost.append(two_pass_cost_avg / lowerbound_cost_avg)
        kl_cost.append(kl_cost_avg / lowerbound_cost_avg)
        distributed_cost.append(dist_cost_avg / lowerbound_cost_avg)
        kale_cost.append(kale_cost_avg / lowerbound_cost_avg)
        chen_etal_cost.append(chen_cost_avg / lowerbound_cost_avg)

    mlist = [i for i in range(len(capacities_list))]

    df = pd.DataFrame({"mlist": mlist, "Two Pass": two_pass_cost, "Distributed": distributed_cost,
                       "Kleindessner et al.": kl_cost, "Kale": kale_cost,"Chen et al.": chen_etal_cost})

    sns.set()
    ax = sns.scatterplot(x='mlist',y='value',hue='Algorithms',style='Algorithms',
                    data=df.melt("mlist",var_name='Algorithms'),legend="full",s=250,markers=['o', 'H', 'v', 'P', '^'])
    capacities_str = []
    for capacities in capacities_list:
        capacities_str.append(",".join(str(c) for c in capacities))

    plt.xticks(mlist,capacities_str)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.xlabel('Capacities' + '\n' + '#repetitions = ' + str(reps))
    plt.ylabel('Algorithm Cost / Opt Lower Bound')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, markerscale=2)
    plt.tight_layout()
    plt.show()
    return

'''
n = 500
eps = 0.1
for n in range(500, 501, 1000):
    print("n= ", n)
    gt = time.time()
    w = 1000.0
    metric_generation = "er", 2.0*np.log(n)/n, w
    g = 2
    random_instance = RandomFairKCenter(n, [1.0 / g]*g, [2]*g, metric_generation)
    gt = time.time() - gt

    startkl = time.time()
    kl_centers = kl.fair_k_center_APPROX(random_instance.d, np.array(random_instance.groups), np.array(random_instance.capacities), np.array([0]))
    kl_cost = random_instance.compute_cost(kl_centers)
    endkl = time.time()

    # startchen = time.time()
    # chen_cost, chen_centers = random_instance.chen_etal_main_linear()
    # endchen = time.time()

    # startchen_bin = time.time()
    # chen_cost_bin, chen_centers_bin = random_instance.chen_etal_main_binary()
    # endchen_bin = time.time()

    startchen_geom = time.time()
    chen_cost_geom, chen_centers_geom = random_instance.chen_etal_main_geometric(eps)
    endchen_geom = time.time()

    startk = time.time()
    kale_cost, kale_centers = random_instance.kale_main(0.1)
    endk = time.time()

    startm = time.time()
    two_pass_cost, two_pass_centers = random_instance.two_pass_main(0.1)
    endm = time.time()

    startdist = time.time()
    ##########################################
    # distributed k-center
    eps = 0.1
    dist_cost,dist_solution,distrtime = random_instance.distributed_kcenter(list(range(random_instance.n)),4,eps)
    ##########################################
    enddist = time.time()

    startdistc = time.time()
    ##########################################
    # distributed k-center
    eps = 0.1
    distc_cost,distc_solution = random_instance.distributed_kcenter_cores(list(range(random_instance.n)),4,eps)
    ##########################################
    enddistc = time.time()


    print("Instance generation time = ", gt, "Rank = ", sum(random_instance.capacities))
    print("Kale cost = ", kale_cost, "time for Kale = ", endk - startk, "#centers", len(kale_centers))
    print("Two-pass cost = ", two_pass_cost, "time for two pass = ", endm - startm, "#centers", len(two_pass_centers))
    print("Kleindessner et al. cost = ", kl_cost, "time = ", endkl - startkl, "#centers", len(kl_centers))
    print("Distributed cost = ", dist_cost, "total time = ", enddist - startdist, "distributed, simulated time = ", distrtime, "#centers = ", len(dist_solution))
    print("Distributed cores cost = ", distc_cost, "total time = ", enddistc - startdistc, "#centers = ", len(distc_solution))
    # print("Chen et al cost = ", chen_cost, "time = ", endchen - startchen, "#centers", len(chen_centers))
    # print("Chen et al cost with binary search = ", chen_cost_bin, "time = ", endchen_bin - startchen_bin, "#centers", len(chen_centers_bin))
    print("Chen et al cost with geometric search = ", chen_cost_geom, "time = ", endchen_geom - startchen_geom, "#centers", len(chen_centers_geom))
'''

# creating a list of instance sizes
nlist = [i for i in range(100, 351, 50)]
# reps: variable for the number repetitions for each value of instance size in nlist
reps = 20
for g in range(5, 6):
    print("g = ", g)
    capacities = [2] * g
    instancesize_runtime_plot(nlist, reps, capacities)
# instance size
n = 500
# creating a list of capacities
capacities_list = [[5, 5, 5], [2, 2, 11], [2, 2, 8, 8], [3, 3, 3, 11], [1, 2, 3, 4, 5], [3, 3, 4, 4, 5],
                   [4, 4, 5, 5, 5, 10], [2, 2, 2, 2, 2, 2]]
# reps: variable for the number repetitions for each set of capacity
reps = 20
approxratio_comparison_plot(n, capacities_list, reps)