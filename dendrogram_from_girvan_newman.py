# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.centrality import girvan_newman


def girvan_newman_partitions(G): #take network graph and return girvan-newman partitions
    # Does G meet the conditions?
    if nx.number_connected_components(G) > 1:
        raise TypeError("Bad graph type: do not use a graph with " + "more connected components")
    _nodes = nx.nodes(G)
    _nn = nx.number_of_nodes(G)
    _good_nodes = np.arange(_nn)
    if not set(_nodes) == set(_good_nodes):
        raise TypeError("Bad graph type: use a graph with nodes " + "which are integers from 0 to (number_of_nodes - 1)")
    _gn_partitions = list(girvan_newman(G))
    gn_partitions = []
    for part in _gn_partitions:
        sorted_part = sorted(part, key=lambda x: min(x))
        gn_partitions.append(sorted_part)
    return gn_partitions


def _list2dict(list_partitions, number_nodes): # take list_partitions and convert it to dictionary
    list_of_dict = [0]*(number_nodes-1)
    c = 0
    for part in list_partitions:
        # Transform to a dict
        tmp_dict = {}
        for i, part_i in enumerate(part):
            for j in part_i:
                tmp_dict[j] = i
        list_of_dict[c] = tmp_dict
        c += 1
    return list_of_dict


def _informative_dict(list_of_dict, number_nodes): #  The dictionary ith contains the information about the partition of the
        # level ith and it has this form:
        #     - key: integer, label of the node;
        #     - value: list of integers,
        #         [label of the community the node belongs to,
        #         distance of the community from the ground level,
                #number of nodes in the community
    _list_of_dict = list_of_dict[::-1]
    informative_dict = [0] * (number_nodes-1)
    # Compute newdict_0, the first dict of the list 'informative_dict'
    dict_0 = dict(_list_of_dict[0])
    newkey_0 = list(dict_0.keys())
    newval_0 = [[i, 0, 1] for i in list(dict_0.values())]
    newdict_0 = {k : v for k, v in zip(newkey_0, newval_0)}
    informative_dict[0] = newdict_0
    # Compute newdict_i
    dict_cur = dict(dict_0)
    c = 0
    for dict_i in _list_of_dict:
        dict_pre = dict(dict_cur)  # Old dict with info about previous partition
        dict_cur = dict(dict_i)    # Old dict with info about current partition
        if c > 0:
            # The new dict with info about the previous partition
            newdict_pre = informative_dict[c-1]
            # Look for key with different val in dict_pre and in dict_cur
            mykey = -999
            for i in range(number_nodes):
                if dict_pre[i] != dict_cur[i]:
                    mykey = i
                    break
            if mykey == -999:
                raise ValueError('ERROR: New community not found,' +
                                 ' fix list_of_dict')
            # Update info of nodes belonging to the new community.
            # Community A and B are the two communities that are joined
            # to form the new community.
            mycom = dict_cur[mykey]
            info_nodeA = newdict_pre[mykey]
            info_nodeB = []
            for k, v in dict_cur.items():
                if (k != mykey) and (v == mycom):
                    if newdict_pre[k] != info_nodeA:
                        info_nodeB = newdict_pre[k]
                        break
            info_newcom = [number_nodes - 1 + c,
                           max(info_nodeA[1], info_nodeB[1]) + 1,
                           info_nodeA[2] + info_nodeB[2]]
            # Create newdict_cur, the new dictionary with info about
            # the current partition
            newdict_cur = dict(newdict_pre)
            comA_pre = info_nodeA[0]
            comB_pre = info_nodeB[0]
            j = 0
            for k, v in newdict_pre.items():
                if (v[0] == comA_pre) or (v[0] == comB_pre):
                    newdict_cur[k] = info_newcom
                    j += 1
                if j == info_newcom[2]:
                    break
            if j < info_newcom[2]:
                raise ValueError('ERROR: not found the %d nodes belonging to' +
                                 ' the new community' % info_newcom[2])
            informative_dict[c] = newdict_cur
        c += 1
    return informative_dict

def agglomerative_matrix(G, list_partitions):
    # Returns
    # -------
    # numpy.ndarray
    #     The "agglomerative matrix" (AM) is a numpy.ndarray with shape
    #     (number_nodes -1)x4. The ith row contains information about the new
    #     community created at that level by merging 2 existing communities.
    #     So the first row contains information about the first new community,
    #     which will contain 2 nodes, while the last row contains information
    #     about the 2 communities which merge to generate the entire graph
    #     (with 'number_nodes' nodes).
    #     The 4 columns contain the following information:
    #         AM[i,0] = integer, label of the existing community A. The community
    #                   A will be merge to the existing community B to form the
    #                   ith new community.
    #         AM[i,1] = integer, label of the existing community B.
    #                   (With AM[i,0] < AM[i,1]).
    #         AM[i,2] = integer, distance of the ith new community from the
    #                   level zero.
    #         AM[i,3] = integer, number of nodes in the ith new community.
    #     A community with an index less than 'number_nodes' corresponds to one
    #     which contains only that single node. A community with an index
    #     belonging to [number_nodes, 2*number_nodes - 3] corresponds to one of
    #     the new communities formed in the agglomeration process.
    # Does G meet the conditions?
    if nx.number_connected_components(G) > 1:
        raise TypeError("Bad graph type: do not use a graph with more " +
                        " connected components")
    _nodes = nx.nodes(G)
    nn = nx.number_of_nodes(G)
    _good_nodes = np.arange(nn)
    if not set(_nodes) == set(_good_nodes):
        raise TypeError("Bad graph type: use a graph with nodes which are " +
                        "integers from 0 to (number_of_nodes - 1)")

    # Set out the list of partitions in a list of dictionaries containing
    # information on the agglomeration of communities.
    list_of_dict = _list2dict(list_partitions, nn)
    list_info_dict = _informative_dict(list_of_dict, nn)
    # Create the 'agglomerative matrix'
    AM = np.zeros((nn - 1, 4), dtype='float')
    row = 0
    comA = 0
    comB = 0
    dist = 0
    nn_com = 0
    for row in range(nn - 1):
        # For row from 0 to nn-2
        if row < nn-2:
            # dict of info about previous partition and current partition
            dict_pre = dict(list_info_dict[row])
            dict_cur = dict(list_info_dict[row + 1])
            # Which are the nodes who belong to the new community?
            new_com = nn + row   # Label of community created at this level
            n_found = 0
            number_nodes = -1
            nodes_in_newcom = []
            for i in range(nn):                # Look for all nodes belonging
                if dict_cur[i][0] == new_com:  # to the new community. Their
                    nodes_in_newcom.append(i)  # labels will be stored in
                    if n_found == 0:           # 'nodes_in_newcom'
                        info_newcom = dict_cur[i]
                        number_nodes = info_newcom[2]
                    n_found += 1
                if n_found == number_nodes:
                    break
            # Look for info of communities A and B which are merged to form
            # the new community
            if number_nodes < 2:
                raise ValueError('ERROR: the new community has less ' +
                                 'than 2 nodes')
            elif number_nodes == 2:
                comA = min(nodes_in_newcom[0], nodes_in_newcom[1])
                comB = max(nodes_in_newcom[0], nodes_in_newcom[1])
                dist = info_newcom[1]
                nn_com = number_nodes
            else:
                tmp_comA = dict_pre[nodes_in_newcom[0]][0]
                for i in range(number_nodes):
                    if dict_pre[nodes_in_newcom[i + 1]][0] != tmp_comA:
                        tmp_comB = dict_pre[nodes_in_newcom[i + 1]][0]
                        break
                comA = min(tmp_comA, tmp_comB)
                comB = max(tmp_comA, tmp_comB)
                dist = info_newcom[1]
                nn_com = number_nodes
            # Fill the agglomerative matrix
            AM[row] = [comA, comB, dist, nn_com]
        # For row number nn-2, the last one. (Rows go from 0 to nn-2)
        if row == nn-2:
            dict_pre = dict(list_info_dict[row])
            info_comA = dict_pre[nn - 1]
            info_comB = []
            for i in range(nn):
                if dict_pre[i][0] != info_comA[0]:
                    info_comB = dict_pre[i]
                    break
            comA = min(info_comA[0], info_comB[0])
            comB = max(info_comA[0], info_comB[0])
            dist = max(info_comA[1], info_comB[1]) + 1
            nn_com = info_comA[2] + info_comB[2]
            # Check: does the last community contain all the nodes?
            if nn_com != nn:
                raise ValueError('ERROR: the last community (which is' +
                                 ' the entire graph) does not contain' +
                                 ' "number_nodes" nodes')
            # Fill the last row of the 'agglomerative_matrix'
            AM[row] = [comA, comB, dist, nn_com]
    return AM


def girvan_newman_best_partition(G, list_partitions):
    # Does G meet the conditions?
    if nx.number_connected_components(G) > 1:
        raise TypeError("Bad graph type: do not use a graph with more" +
                        " connected components")
    _nodes = nx.nodes(G)
    nn = nx.number_of_nodes(G)
    _good_nodes = np.arange(nn)
    if not set(_nodes) == set(_good_nodes):
        raise TypeError("Bad graph type: use a graph with nodes which" +
                        " are integers from 0 to (number_of_nodes - 1)")
    # Look for the best partition
    best_partition = []
    MAX_mod = -99
    c = 0
    for part in list_partitions:
        # Compute modularity
        tmp_mod = modularity(G, part)
        # If modularity icreases, then update `best_partition`
        if tmp_mod > MAX_mod:
            MAX_mod = tmp_mod
            best_partition = part
            id_best_part = c
        c += 1
    return (best_partition, id_best_part)


def distance_of_partition(agglomerative_matrix, n_communities):
    
    # Check if 'n_communities' belongs to the interval [1, number_nodes].
    nn = len(agglomerative_matrix[:, 0]) + 1
    if (n_communities < 1) or (n_communities > nn):
        raise TypeError('Bad number of communities: n_communities must be' +
                        ' an integer between 1 and number_nodes')
    # High of the level of the hierarchy in which the graph is split
    # into 'n_communities' different partitions.
    high_max = int(agglomerative_matrix[-1, 2])
    partition_height = high_max - (n_communities - 2)

    return partition_height