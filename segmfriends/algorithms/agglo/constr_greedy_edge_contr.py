import numpy as np
from nifty.graph import agglo as nagglo

import segmfriends.features.mappings

NODE_KEYS = {'node_sizes':0,
             'node_labels': 1,
             'node_GT': 2}
EDGE_KEYS = {'edge_sizes': 0,
             'edge_indicators': 1,
             'dendrogram_heigh': 2,
             'merge_times': 3,
             'loss_targets': 4,
             'loss_weights': 5}


class constrained_fixation_clustering(object):
    def __init__(self, graph, edge_sizes, node_sizes,
                 image_shape,
                 isLocalEdge,
                 GT_labels_nodes=None,
                 ignore_label=None,
                 constrained=True, compute_loss_data=True,
                 max_nb_milesteps=-1,
                 p0=1.0, p1=1.0, zeroInit=False, weight_mistakes=1.0, weight_successes=1.0):
        '''
            * ignore_label: label in the passed GT_labels that should be ignored.
                            If a merge involves the ignore_label it is performed, but not included in the training.
                            The resulting merged segment becomes labels as "ignore_label" (the label spreads).

            * if computeLossData==False, expensive backtracking of edge-union-find is avoided

        '''
        if ignore_label is None:
            # ignore_label = np.uint64(-1)
            ignore_label = -1

        self._graph = graph
        self._image_shape = image_shape
        self._nb_nodes = nb_nodes = graph.numberOfNodes
        self._nb_edges = nb_edges = graph.numberOfEdges
        self._constrained = constrained
        self._compute_loss_data = compute_loss_data
        self._max_nb_milesteps = max_nb_milesteps
        self.isOver = False
        self.iterations_milesteps = []
        self.current_data = {}
        self.initial_edge_sizes = edge_sizes
        self.initial_node_sizes = node_sizes


        if GT_labels_nodes is None:
            compute_loss_data = False
            constrained = False
            GT_labels_nodes = np.zeros(nb_nodes, dtype=np.int)

        self._constrained_clustering = nagglo.constrainedGeneralizedFixationClustering(graph=graph, isLocalEdge=isLocalEdge.astype(np.float),
                                                        p0=p0, p1=p1, zeroInit=zeroInit, weight_mistakes=weight_mistakes,
                                                        weight_successes=weight_successes,
                                                        edgeSizes=edge_sizes, nodeSizes=node_sizes,
                                                        GTlabels=GT_labels_nodes, ignore_label=ignore_label,
                                                        constrained=constrained, computeLossData=compute_loss_data,
                                                        verbose=False)


    def run_next_milestep(self, edge_affs, nb_iterations=-1):
        """
            * affinities: 1 if merge, 0 otherwise
            * nb_iterations: with -1 runs until threshold is reached
            * return True if the agglomeration is over
        """
        if self.nb_performed_milesteps>=self._max_nb_milesteps-1:
            # print("Max number of milesteps reached. Running until termination.")
            nb_iterations = -1

        self.isOver = self._constrained_clustering.runNextMilestep(nb_iterations_in_milestep=nb_iterations,
                                                           new_merge_prios=edge_affs,
                                                           new_not_merge_prios=1. - edge_affs)

        self.iterations_milesteps.append(self._constrained_clustering.time())

        # TODO: do it better... (save data of every iteration?)
        # Collect and store data milestep:
        data_milestep = self._constrained_clustering.collectDataMilestep()
        node_sizes, node_labels, node_GT, edge_sizes, edge_merge_prios, dendrogram_heigh, \
        merge_times, loss_targets, loss_weights = data_milestep

        self.current_data['node_sizes'] = node_sizes
        self.current_data['edge_sizes'] = edge_sizes
        self.current_data['node_labels']= node_labels
        self.current_data['node_GT']= node_GT
        self.current_data['edge_merge_prios'] = edge_merge_prios
        self.current_data['dendrogram_heigh'] = dendrogram_heigh
        self.current_data['merge_times'] = merge_times
        if self._compute_loss_data:
            self.current_data['loss_targets'] = loss_targets
            self.current_data['loss_weights'] = loss_weights


        return self.isOver

    @ property
    def nb_performed_milesteps(self):
        return len(self.iterations_milesteps)

    @property
    def nb_performed_iterations(self):
        if len(self.iterations_milesteps)!=0:
            return self.iterations_milesteps[-1]
        else:
            return 0

    def get_all_last_data_milestep(self):
        """
        Merge together all the node and edge features.

        Returned values (all float for every edge/node of the original rag):

            * node_sizes
            * node_labels
            * node_GT_labels (ignore label is propagated to the merged nodes)

            * edge_sizes            (-1.0. if contracted)
            * edge_indicators       (-1.0  if contracted)
            * dendrogram_heigh      (-1.0  if not contracted)
            * merge_times           (-1.0  if not contracted)
            * loss_targets          (not returned if not computeLossData, + 1.0 should be merged, - 1.0 should not be merged)
            * loss_weights          (not returned if not computeLossData)

        """
        last_data = self.current_data

        node_features = np.stack([last_data['node_sizes'], last_data['node_labels'], last_data['node_GT']], axis=-1)
        edge_features = np.stack([last_data['edge_sizes'], last_data['edge_merge_prios'], last_data['dendrogram_heigh'], last_data['merge_times']], axis=-1)
        if self._compute_loss_data:
            edge_features_loss = np.stack([last_data['loss_targets'],
                                          last_data['loss_weights']], axis=-1)
            edge_features =  np.concatenate([edge_features, edge_features_loss], axis=-1)

        return node_features, edge_features


    def current_segmentation(self, init_segm, n_threads=1):
        """Change this please..."""
        return segmfriends.features.mappings.map_features_to_label_array(
            init_segm,
            np.expand_dims(self.current_data['node_labels'], axis=-1),
            number_of_threads=n_threads
        )[..., 0]