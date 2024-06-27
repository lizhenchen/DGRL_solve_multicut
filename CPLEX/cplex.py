import pickle
import time
import networkx as nx
import numpy as np
import csv
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


def multicut_ilp(graph, costs, time_limit=None, **kwargs):
    """ Solve multicut problem with ilp solver.
    Introduced in "Globally Optimal Closed-surface Segmentation for Connectomics":
    https://link.springer.com/chapter/10.1007/978-3-642-33712-3_56
    Requires nifty build with CPLEX, GUROBI or GLPK.
    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
    """
    if not any((nifty.Configuration.WITH_CPLEX, nifty.Configuration.WITH_GLPK, nifty.Configuration.WITH_GUROBI)):
        raise RuntimeError("multicut_ilp requires nifty built with at least one of CPLEX, GLPK or GUROBI")
    objective = _to_objective(graph, costs)
    solver = objective.multicutIlpFactory().create(objective)
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


def run_cplex(dataset_nx, dataset_ny, cost, limit):

    obj = []
    for i in range(len(dataset_nx)):
        print(i)
        node_labels = multicut_ilp(dataset_ny[i], cost[i], time_limit=limit).tolist()
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
    # MODEL_NAME = 'realworld/modularity-clustering/lesmis'
    MODEL_NAME = 'ER-40'
    print(MODEL_NAME)

    dataset_nx, dataset_ny, costs = nx2ny(MODEL_NAME, 'test_nx_graphs')
    print('LEN:', len(dataset_nx))

    csvfile = open('../%s/cplex_results_without_limit.csv' % MODEL_NAME, 'w', newline='')

    writer = csv.writer(csvfile)

    start = time.time()
    # obj_cplex = run_cplex(dataset_nx, dataset_ny, costs, limit=1800)
    obj_cplex = run_cplex(dataset_nx, dataset_ny, costs, limit=None)
    end = time.time()

    writer.writerow(['CPLEX'] + obj_cplex + [''] + [''] + [str(end - start)] + [sum(obj_cplex)])

    csvfile.close()
