import pickle
import networkx as nx
import numpy as np
import csv
import time
import random
import nifty
import nifty.graph.opt.multicut as nmc


def _to_objective(graph, costs):

    if isinstance(graph, nifty.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
    objective = nmc.multicutObjective(graph_, costs)

    return objective


def _get_solver_factory(objective, internal_solver, warmstart=True, warmstart_kl=False):

    if internal_solver == 'kernighan-lin':
        sub_solver = objective.kernighanLinFactory(warmStartGreedy=warmstart)
    elif internal_solver == 'greedy-additive':
        sub_solver = objective.greedyAdditiveFactory()
    elif internal_solver == 'greedy-fixation':
        sub_solver = objective.greedyFixationFactory()
    elif internal_solver == 'cut-glue-cut':
        if not nifty.Configuration.WITH_QPBO:
            raise RuntimeError("multicut_cgc requires nifty built with QPBO")
        sub_solver = objective.cgcFactory(warmStartGreedy=warmstart, warmStartKl=warmstart_kl)
    elif internal_solver == 'ilp':
        if not any((nifty.Configuration.WITH_CPLEX, nifty.Configuration.WITH_GLPK, nifty.Configuration.WITH_GUROBI)):
            raise RuntimeError("multicut_ilp requires nifty built with at least one of CPLEX, GLPK or GUROBI")
        sub_solver = objective.multicutIlpFactory()
    elif internal_solver in ('fusion-move', 'decomposition'):
        # raise NotImplementedError(f"Using {internal_solver} as internal solver is currently not supported.")
        pass
    else:
        # raise ValueError(f"{internal_solver} cannot be used as internal solver.")
        pass
    return sub_solver


def _get_visitor(objective, time_limit=None, **kwargs):

    logging_interval = kwargs.pop('logging_interval', None)
    log_level = kwargs.pop('log_level', 'INFO')
    if time_limit is not None or logging_interval is not None:
        logging_interval = int(np.iinfo('int32').max) if logging_interval is None else logging_interval
        time_limit = float('inf') if time_limit is None else time_limit
        log_level = getattr(nifty.LogLevel, log_level, nifty.LogLevel.INFO)

        # I can't see a real difference between loggingVisitor and verboseVisitor.
        # Use loggingVisitor for now.

        # visitor = objective.verboseVisitor(visitNth=logging_interval,
        #                                    timeLimitTotal=time_limit,
        #                                    logLevel=log_level)
        visitor = objective.loggingVisitor(visitNth=logging_interval,
                                           timeLimitTotal=time_limit,
                                           logLevel=log_level)
        return visitor
    else:
        return None


def multicut_gaec(graph, costs, time_limit=None, **kwargs):
    """ Solve multicut problem with greedy-addtive edge contraction solver.
    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf
    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
    """
    objective = _to_objective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)

    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_kernighan_lin(graph, costs, time_limit=None, warmstart=True, **kwargs):
    """ Solve multicut problem with kernighan lin solver.
    Introduced in "An efficient heuristic procedure for partitioning graphs":
    http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf
    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
    """
    objective = _to_objective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)

    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_cgc(graph, costs, time_limit=None, warmstart=True, **kwargs):
    """ Solve multicut problem with cut,glue&cut solver.
    Introduced in "Cut, Glue & Cut: A Fast, Approximate Solver for Multicut Partitioning":
    https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Beier_Cut_Glue__2014_CVPR_paper.html
    Requires nifty build with QPBO.
    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
        warmstart_kl [bool] - also use kernighan lin to warmstart (default: True)
    """
    if not nifty.Configuration.WITH_QPBO:
        raise RuntimeError("multicut_cgc requires nifty built with QPBO")
    objective = _to_objective(graph, costs)
    solver = objective.cgcFactory(warmStartGreedy=warmstart).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)

    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_fusion_moves(graph, costs, time_limit=None, n_threads=1,
                          internal_solver='kernighan-lin',
                          warmstart=True,
                          seed_fraction=.05, num_it=1000, num_it_stop=25, sigma=2.,
                          **kwargs):
    """ Solve multicut problem with fusion moves solver.
    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf
    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
        n_threasd [int] - number of threads (default: 1)
        internal_solver [str] - name of solver used for connected components
            (default: 'kernighan-lin')
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
        warmstart_kl [bool] - also use kernighan lin to warmstart (default: True)
        seed_fraction [float] - fraction of nodes used as seeds for proposal generation
            (default: .05)
        num_it [int] - maximal number of iterations (default: 1000)
        num_it_stop [int] - stop if no improvement after num_it_stop (default: 1000)
        sigma [float] - smoothing factor for weights in proposal generator (default: 2.)
    """
    objective = _to_objective(graph, costs)
    int_solver = _get_solver_factory(objective, internal_solver)
    sub_solver = objective.fusionMoveSettings(mcFactory=int_solver)
    proposal_gen = objective.watershedCcProposals(sigma=sigma, numberOfSeeds=seed_fraction)

    solver = objective.ccFusionMoveBasedFactory(fusionMove=sub_solver,
                                                warmStartGreedy=warmstart,
                                                proposalGenerator=proposal_gen,
                                                numberOfThreads=n_threads,
                                                numberOfIterations=num_it,
                                                stopIfNoImprovement=num_it_stop).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)

    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def nx2ny(model_name, test_name):

    test_original_graphs = pickle.load(open('../%s/%s.pkl' % (model_name, test_name), 'rb'))
    dataset_nx = test_original_graphs
    dataset_ny = []
    costs = [None] * len(dataset_nx)
    i = 0
    for graph_nx in dataset_nx:
        num_node = nx.number_of_nodes(graph_nx)
        graph_ny = nifty.graph.undirectedGraph(num_node)
        cost = []
        for edge in graph_nx.edges:
            graph_ny.insertEdge(edge[0], edge[1])
            cost.append(graph_nx.edges[edge]['weight'])
        dataset_ny.append(graph_ny)
        costs[i] = np.array(cost)
        i = i + 1

    return dataset_nx, dataset_ny, costs


class MulticutEnv():

    def __init__(self, graph_original_nx):

        self.cg = graph_original_nx
        for i in range(len(self.cg)):
            self.cg.nodes[i]['contain'] = {i}

    def objective(self):

        weight_array = np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)])

        return sum(weight_array)

    def best(self):

        num_cluster = 0
        clusters = []
        for i in self.cg.nodes:
            quality = []
            quality_max = -1e5
            for k in range(num_cluster):
                quality_k = []
                for j in clusters[k]:
                    if self.cg.has_edge(i, j):
                        quality_k.append(self.cg.edges[(i, j)]['weight'])
                if len(quality_k) != 0:
                    quality.append(max(quality_k))
                else:
                    quality.append(-1e5)
            if len(quality) != 0:
                quality_max = max(quality)
            if quality_max > 0:
                k_max = quality.index(quality_max)
                clusters[k_max] = clusters[k_max] | {i}
            else:
                num_cluster += 1
                clusters.append({i})

        w_inside = 0
        for k in range(len(clusters)):
            for i in clusters[k]:
                for j in clusters[k]:
                    if self.cg.has_edge(i, j) and i > j:
                        w_inside += self.cg.edges[(i, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))
        object_fun = w_all - w_inside

        return object_fun

    def first(self):

        num_cluster = 0
        clusters = []
        for i in self.cg.nodes:
            quality = []
            quality_max = -1e5
            for k in range(num_cluster):
                quality_k = []
                for j in clusters[k]:
                    if self.cg.has_edge(i, j) and self.cg.edges[(i, j)]['weight'] > 0:
                        quality_k.append(j)
                if len(quality_k) != 0:
                    quality.append(max(quality_k))
                else:
                    quality.append(-1e5)
            if len(quality) != 0:
                quality_max = max(quality)
            if quality_max > 0:
                k_max = quality.index(quality_max)
                clusters[k_max] = clusters[k_max] | {i}
            else:
                num_cluster += 1
                clusters.append({i})

        w_inside = 0
        for k in range(len(clusters)):
            for i in clusters[k]:
                for j in clusters[k]:
                    if self.cg.has_edge(i, j) and i > j:
                        w_inside += self.cg.edges[(i, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))
        object_fun = w_all - w_inside

        return object_fun

    def vote(self):

        num_cluster = 0
        clusters = []
        for i in self.cg.nodes:
            quality = []
            quality_max = -1e5
            for k in range(num_cluster):
                quality_k = []
                for j in clusters[k]:
                    if self.cg.has_edge(i, j):
                        quality_k.append(self.cg.edges[(i, j)]['weight'])
                if len(quality_k) != 0:
                    quality.append(sum(quality_k))
                else:
                    quality.append(-1e5)
            if len(quality) != 0:
                quality_max = max(quality)
            if quality_max > 0:
                k_max = quality.index(quality_max)
                clusters[k_max] = clusters[k_max] | {i}
            else:
                num_cluster += 1
                clusters.append({i})

        w_inside = 0
        for k in range(len(clusters)):
            for i in clusters[k]:
                for j in clusters[k]:
                    if self.cg.has_edge(i, j) and i > j:
                        w_inside += self.cg.edges[(i, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))
        object_fun = w_all - w_inside

        return object_fun

    def pivot(self):

        num_cluster = 0
        clustered_all = set()
        clusters = []
        for i in self.cg.nodes:
            if i not in clustered_all:
                cluster_part = set()
                num_cluster += 1
                for j in range(i + 1, self.cg.number_of_nodes()):
                    if self.cg.has_edge(i, j) and (j not in clustered_all):
                        if self.cg.edges[i, j]['weight'] > 0:
                            cluster_part = cluster_part | {j}
                clusters.append({i} | cluster_part)
            clustered_all = clustered_all | (clusters[-1])

        w_inside = 0
        for k in range(len(clusters)):
            for i in clusters[k]:
                for j in clusters[k]:
                    if self.cg.has_edge(i, j) and i > j:
                        w_inside += self.cg.edges[(i, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))
        object_fun = w_all - w_inside

        return object_fun, clusters

    def bec(self):

        num_nodes = self.cg.number_of_nodes()
        for t in range(num_nodes - 1):
            weights = [float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]
            edges = [[u, v] for (u, v, w) in self.cg.edges(data=True)]
            size = [len(self.cg.nodes[edge[0]]['contain']) + len(self.cg.nodes[edge[1]]['contain']) for edge in edges]
            weights = [weights[i]/size[i] for i in range(len(size))]
            max_weight = max(weights)
            if max_weight <= 0.:
                break
            max_weight_index = weights.index(max_weight)
            u, v = edges[max_weight_index][0], edges[max_weight_index][1]

            H = self.cg.copy()

            new_edges = [(u, w) for x, w, d in self.cg.edges(v, data=True) if w != u]
            v_data_contain = self.cg.nodes[v]['contain']

            H.remove_node(v)
            H.nodes[u]['contain'] = H.nodes[u]['contain'] | v_data_contain

            new_edges_weights = []
            for e in new_edges:
                if self.cg.has_edge(e[0], e[1]):
                    new_edges_weights.append((e[0], e[1], {'weight': self.cg.edges[(u, e[1])]['weight'] + self.cg.edges[(v, e[1])]['weight']}))
                else:
                    new_edges_weights.append((e[0], e[1], {'weight': self.cg.edges[(v, e[1])]['weight']}))
            H.add_edges_from(new_edges_weights)
            self.cg = H

        object_fun = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))

        return object_fun

    def gf(self):

        original_weights = [float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]
        edges = [[u, v] for (u, v, w) in self.cg.edges(data=True)]
        abs_original = [abs(i) for i in original_weights]
        sorted_abs = sorted(abs_original, reverse=True)
        for e in self.cg.edges():
            self.cg.edges[e]['fixation'] = False

        for i in range(len(sorted_abs)):
            if self.cg.number_of_edges() == 0:
                break
            fix = []
            for e in self.cg.edges():
                fix.append(self.cg.edges[e]['fixation'])
            if sum(fix) == len(fix):
                break
            edge_index = abs_original.index(sorted_abs[i])
            abs_original[edge_index] = 1e9
            edge_weight = original_weights[edge_index]
            u_, v_ = edges[edge_index][0], edges[edge_index][1]
            d = {}
            for j in self.cg.nodes():
                contains = self.cg.nodes[j]['contain']
                for k in contains:
                    d[k] = j
            u, v = d[u_], d[v_]
            assert u_ in self.cg.nodes[u]['contain']
            assert v_ in self.cg.nodes[v]['contain']
            if u == v:
                continue
            if self.cg.edges[(u, v)]['fixation'] is True:
                continue

            if edge_weight > 0:
                H = self.cg.copy()
                new_edges = [(u, w) for x, w, d in self.cg.edges(v, data=True) if w != u]
                v_data_contain = self.cg.nodes[v]['contain']
                H.remove_node(v)
                H.nodes[u]['contain'] = H.nodes[u]['contain'] | v_data_contain
                new_edges_weights = []
                for e in new_edges:
                    if self.cg.has_edge(e[0], e[1]):
                        new_edges_weights.append((e[0], e[1], {
                            'weight': self.cg.edges[(u, e[1])]['weight'] + self.cg.edges[(v, e[1])]['weight'],
                            'fixation': bool(self.cg.edges[(u, e[1])]['fixation'] + self.cg.edges[(v, e[1])]['fixation'])}))
                    else:
                        new_edges_weights.append((e[0], e[1], {'weight': self.cg.edges[(v, e[1])]['weight'], 'fixation': self.cg.edges[(v, e[1])]['fixation']}))
                H.add_edges_from(new_edges_weights)
                self.cg = H
            elif edge_weight < 0:
                self.cg.edges[(u, v)]['fixation'] = True
            else:
                print('edge_weight == 0.')
                break

        object_fun = sum(np.array([float(w['weight']) for (u, v, w) in self.cg.edges(data=True)]))

        return object_fun


def calculate_obj(graph, cluster_list):

    w_inside = 0
    for k in range(len(cluster_list)):
        for i in cluster_list[k]:
            for j in cluster_list[k]:
                if graph.has_edge(i, j) and i > j:
                    w_inside += graph.edges[(i, j)]['weight']
    w_all = sum(np.array([float(w['weight']) for (u, v, w) in graph.edges(data=True)]))
    object_fun = w_all - w_inside

    return object_fun


def run_best(MulticutEnv, dataset):

    obj_final_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value = env.best()
        obj_final_list.append(obj_value)

    return obj_final_list


def run_first(MulticutEnv, dataset):

    obj_final_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value = env.first()
        obj_final_list.append(obj_value)

    return obj_final_list


def run_vote(MulticutEnv, dataset):

    obj_final_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value = env.vote()
        obj_final_list.append(obj_value)

    return obj_final_list


def run_pivot(MulticutEnv, dataset):

    obj_final_list = []
    cluster_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value, clusters = env.pivot()
        obj_final_list.append(obj_value)
        cluster_list.append(clusters)

    return obj_final_list, cluster_list


def run_bec(MulticutEnv, dataset):

    obj_final_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value = env.bec()
        obj_final_list.append(obj_value)

    return obj_final_list


def run_gf(MulticutEnv, dataset):

    obj_final_list = []
    for i in range(len(dataset)):
        env = MulticutEnv(dataset[i])
        obj_value = env.gf()
        obj_final_list.append(obj_value)

    return obj_final_list


def run_gaec(dataset_nx, dataset_ny, cost):

    obj = []
    for i in range(len(dataset_nx)):
        node_labels = multicut_gaec(dataset_ny[i], cost[i]).tolist()
        from collections import defaultdict
        dd = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(node_labels)]:
            dd[k].append(va)

        w_inside = 0
        for k in dd.keys():
            for q in dd[k]:
                for j in dd[k]:
                    if dataset_nx[i].has_edge(q, j) and q > j:
                        w_inside += dataset_nx[i].edges[(q, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in dataset_nx[i].edges(data=True)]))
        object_fun = w_all - w_inside
        obj.append(object_fun)

    return obj


def run_klj(dataset_nx, dataset_ny, cost):

    obj = []
    for i in range(len(dataset_nx)):
        node_labels = multicut_kernighan_lin(dataset_ny[i], cost[i], warmstart=True).tolist()
        from collections import defaultdict
        dd = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(node_labels)]:
            dd[k].append(va)

        w_inside = 0
        for k in dd.keys():
            for q in dd[k]:
                for j in dd[k]:
                    if dataset_nx[i].has_edge(q, j) and q > j:
                        w_inside += dataset_nx[i].edges[(q, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in dataset_nx[i].edges(data=True)]))
        object_fun = w_all - w_inside
        obj.append(object_fun)

    return obj


def run_cgc(dataset_nx, dataset_ny, cost):

    obj = []
    for i in range(len(dataset_nx)):
        node_labels = multicut_cgc(dataset_ny[i], cost[i]).tolist()
        from collections import defaultdict
        dd = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(node_labels)]:
            dd[k].append(va)

        w_inside = 0
        for k in dd.keys():
            for q in dd[k]:
                for j in dd[k]:
                    if dataset_nx[i].has_edge(q, j) and q > j:
                        w_inside += dataset_nx[i].edges[(q, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in dataset_nx[i].edges(data=True)]))
        object_fun = w_all - w_inside
        obj.append(object_fun)

    return obj


def run_fm(dataset_nx, dataset_ny, cost, internal='kernighan-lin'):  # 'greedy-additive'

    obj = []
    for i in range(len(dataset_nx)):
        node_labels = multicut_fusion_moves(dataset_ny[i], cost[i], internal_solver=internal).tolist()
        from collections import defaultdict
        dd = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(node_labels)]:
            dd[k].append(va)

        w_inside = 0
        for k in dd.keys():
            for q in dd[k]:
                for j in dd[k]:
                    if dataset_nx[i].has_edge(q, j) and q > j:
                        w_inside += dataset_nx[i].edges[(q, j)]['weight']
        w_all = sum(np.array([float(w['weight']) for (u, v, w) in dataset_nx[i].edges(data=True)]))
        object_fun = w_all - w_inside
        obj.append(object_fun)

    return obj


if __name__ == '__main__':

    # MODEL_NAME = 'BA-60-20000'
    # MODEL_NAME = 'realworld/image-seg/image-seg'
    # MODEL_NAME = 'realworld/knott-3d-150/knott-3d-150'
    # MODEL_NAME = 'realworld/modularity-clustering/polbooks'
    # MODEL_NAME = 'larger/WS-100'
    MODEL_NAME = 'ER-40'

    print(MODEL_NAME)

    # NUM_BOEM_ITERATION = 100

    dataset_nx, dataset_ny, costs = nx2ny(MODEL_NAME, 'test_nx_graphs')

    csvfile = open('../%s/baseline_results.csv' % MODEL_NAME, 'w', newline='')
    writer = csv.writer(csvfile)

    writer.writerow(['Initial'] + [sum(i) for i in costs])

    time_1 = time.time()
    obj_cgc = run_cgc(dataset_nx, dataset_ny, costs)
    time_2 = time.time()
    obj_fm = run_fm(dataset_nx, dataset_ny, costs, internal='kernighan-lin')
    print(sum(obj_fm))
    time_3 = time.time()
    obj_fm_1 = run_fm(dataset_nx, dataset_ny, costs, internal='greedy-additive')
    print(sum(obj_fm_1))
    time_3_1 = time.time()
    obj_klj = run_klj(dataset_nx, dataset_ny, costs)
    time_4 = time.time()
    obj_best = run_best(MulticutEnv, dataset_nx)
    time_5 = time.time()
    obj_first = run_first(MulticutEnv, dataset_nx)
    time_6 = time.time()
    obj_vote = run_vote(MulticutEnv, dataset_nx)
    time_7 = time.time()
    obj_pivot, _ = run_pivot(MulticutEnv, dataset_nx)
    time_8 = time.time()
    obj_gaec = run_gaec(dataset_nx, dataset_ny, costs)
    time_9 = time.time()
    obj_gf = run_gf(MulticutEnv, dataset_nx)
    time_10 = time.time()
    obj_bec = run_bec(MulticutEnv, dataset_nx)
    time_11 = time.time()

    writer.writerow(['CGC'] + obj_cgc + [''] + [''] + [str(time_2 - time_1)] + [sum(obj_cgc)])
    writer.writerow(['FM_kl'] + obj_fm + [''] + [''] + [str(time_3 - time_2)] + [sum(obj_fm)])
    writer.writerow(['FM_gaec'] + obj_fm_1 + [''] + [''] + [str(time_3_1 - time_3)] + [sum(obj_fm_1)])
    writer.writerow(['KLj'] + obj_klj + [''] + [''] + [str(time_4 - time_3_1)] + [sum(obj_klj)])
    writer.writerow(['BEST'] + obj_best + [''] + [''] + [str(time_5 - time_4)] + [sum(obj_best)])
    writer.writerow(['FIRST'] + obj_first + [''] + [''] + [str(time_6 - time_5)] + [sum(obj_first)])
    writer.writerow(['VOTE'] + obj_vote + [''] + [''] + [str(time_7 - time_6)] + [sum(obj_vote)])
    writer.writerow(['PIVOT'] + obj_pivot + [''] + [''] + [str(time_8 - time_7)] + [sum(obj_pivot)])
    writer.writerow(['GAEC'] + obj_gaec + [''] + [''] + [str(time_9 - time_8)] + [sum(obj_gaec)])
    writer.writerow(['GF'] + obj_gf + [''] + [''] + [str(time_10 - time_9)] + [sum(obj_gf)])
    writer.writerow(['BEC'] + obj_bec + [''] + [''] + [str(time_11 - time_10)] + [sum(obj_bec)])

    csvfile.close()
