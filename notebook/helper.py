#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import math
import csv
import shutil
import networkx as nx
from networkx.readwrite import json_graph
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import json
from collections import Counter, OrderedDict, defaultdict, Callable
import itertools
from bitsets import bitset  # pip install bitsets

NOTEBOOK_DIR = sys.argv[1]
NOTEBOOK_INPUT_DIR = os.path.join(NOTEBOOK_DIR, 'data/')
NOTEBOOK_OUT_DIR = os.path.join(NOTEBOOK_DIR, 'data/')

# Read labels and
BRAIN_LABELS = []
filename = os.path.join(NOTEBOOK_INPUT_DIR, 'input/brain_labels_68.txt')
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        BRAIN_LABELS.append(row[0])
print('Read: ' + filename)

FUNC_IDS = []
filename = os.path.join(NOTEBOOK_INPUT_DIR, 'input/brain_functional_ids.txt')
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        FUNC_IDS.append(row[0])
print('Read: ' + filename)

filename = os.path.join(NOTEBOOK_INPUT_DIR, 'input/brain_graph_68.txt')
adj = np.loadtxt(open(filename, "rb"), dtype=int, delimiter=',', skiprows=0)
# Remove diagonal elements to have a real adjacency matrix
adj = adj - np.diag(np.diag(adj))
AG = nx.from_numpy_matrix(adj)
print('Read: ' + filename), '\n'
print 'Average ajacency matrix:', adj.shape

FUNC_ZONES = 7
FUNC_NODE_COUT = AG.number_of_nodes()
X_SPACING = 3
Y_SPACING = 5


def create_node_data(layer_id, input_id, weight,
                     brain_lab=BRAIN_LABELS, brain_func_ids=FUNC_IDS):
    data = {}
    data['ts_group_id'] = 0
    data['layer_pos'] = layer_id
    data['input_id'] = input_id
    data['func_id'] = brain_func_ids[input_id - 1]  # 0 to 67 entries
    data['weight'] = weight
    data['x'] = layer_id * X_SPACING
    data['y'] = input_id * Y_SPACING
    data['pos'] = (data['x'], data['y'])

    name = brain_lab[input_id - 1]
    short_name = ''
    brain_side = ''
    if name.startswith('lh_'):
        short_name = name[3:]
        brain_side = 'left'
    elif name.startswith('rh_'):
        short_name = name[3:]
        brain_side = 'right'
    else:
        short_name = name
    data['label'] = short_name
    data['brain_side'] = brain_side
    data['name'] = name

    return data


def build_graph_from_activated_layers(feature, tol, input_graph=AG):
    X = feature.reshape((-1, input_graph.number_of_nodes()))
    G = nx.Graph()
    nb_layers = X.shape[0]
    # Create the domain (input_ids) range
    ids = tuple(np.arange(1, X.shape[1] + 1))
    input_ids = bitset('Graph', ids)

    # Create edges and nodes if needed
    for i in xrange(nb_layers - 1):
        current_nodes = input_ids.frombools(X[i] > tol).members()
        current_nodes += input_ids.frombools(X[i] < -tol).members()
        next_nodes = input_ids.frombools(X[i + 1] > tol).members()
        next_nodes += input_ids.frombools(X[i + 1] < -tol).members()


        for c in current_nodes:
            src = c + (i * len(ids))
            for n in next_nodes:
                tgt = n + ((i + 1) * len(ids))
                # Check that input_id are real adajcent
                # in the global adjacency matrix
                # beware, the global adj matrix starts at 0
                if input_graph.has_edge(c - 1, n - 1) or c == n:

                    if src not in G:
                        data = create_node_data(i, c, X[i, c - 1])
                        G.add_node(src, data)

                    if tgt not in G:
                        data = create_node_data(i + 1, n, X[i + 1, n - 1])
                        G.add_node(tgt, data)

                    G.add_edge(src, tgt)
    return G


def build_graph_from_feature_tuples(X, tol, input_graph=AG):
    G = nx.Graph()
    # the list is ordered by layer and input_id
    nb_layers = X[-1][0][0] + 1
    input_node_count = input_graph.number_of_nodes()

    # Create edges, and crate nodes if needed
    for i in xrange(nb_layers - 1):
        # for each current node
        for c in itertools.ifilter(lambda x: x[0][0] == i
                                   and (x[1] > tol or x[1] < -tol), X):
            input_id = c[0][1] + 1
            src = input_id + (i * input_node_count)
            for n in itertools.ifilter(lambda x: x[0][0] == i + 1
                                       and (x[1] > tol or x[1] < -tol), X):

                tgt_in_id = n[0][1] + 1
                tgt = tgt_in_id + ((i + 1) * input_node_count)
                # Check that input_id are real adajcent in the global
                # adjacency matrix. Beware, ids in adj matrix starts at 0
                if input_graph.has_edge(input_id - 1, tgt_in_id - 1) or input_id == tgt_in_id:

                    if src not in G:
                        data = create_node_data(i, input_id, c[1])
                        G.add_node(src, data)

                    if tgt not in G:
                        data = create_node_data(i + 1, tgt_in_id, n[1])
                        G.add_node(tgt, data)

                    G.add_edge(src, tgt)

    return G


def dump_components(method_name, components, overrideFolder=True):
    folderpath = os.path.join(NOTEBOOK_OUT_DIR, method_name)
    if not os.path.exists(folderpath):  # create folder is needed
        os.makedirs(folderpath)
    else:
        if overrideFolder:
            shutil.rmtree(folderpath)  # delete content
            os.makedirs(folderpath)

    for c in components:
        if c.g.number_of_nodes() > 0:
            fullpath = os.path.join(folderpath,
                                    'component_' + str(c.id) + '.json')
            with open(fullpath, 'w') as f:
                json.dump(json_graph.node_link_data(c.g), f)
    return folderpath


def export_components_to_matlab(method_name, components, overrideFolder=False):
    folderpath = os.path.join(NOTEBOOK_OUT_DIR, method_name)
    if not os.path.exists(folderpath):  # create folder is needed
        os.makedirs(folderpath)
    else:
        if overrideFolder:
            shutil.rmtree(folderpath)  # delete content
            os.makedirs(folderpath)

    path = os.path.join(folderpath, method_name + '_components')
    out = {}
    for c in components:
        plist = []
        patterns = c.patterns()
        if patterns:
            for p in patterns:
                a = np.zeros((FUNC_NODE_COUT,), dtype=np.int8)
                for k, v in p.input_ids.iteritems():
                    a[k-1] = v
                plist.append(a)

            out['component_' + str(c.id)] = np.array(plist)

    sp.io.savemat(path, out)
    return path


def draw_and_save_patterns(method_name, components):
    print 'Wrote json components to:', dump_components(method_name, components)
    print 'Wrote', export_components_to_matlab(method_name, components)
    for i, c in enumerate(components):
        f = c.draw()
        if f:
            f.set_visible(True)
            f.savefig(os.path.join(NOTEBOOK_OUT_DIR, method_name + '/' +
                                   method_name + '_' + str(i) +
                                   '.png'))


def rebuild_component_from_cluster(comp_id, comp_list, tol):
    """Extract average component from a list of
    component in the same cluster
    """
    # Histogram
    counter = defaultdict(int)
    for c in comp_list:
        for pair in c.features():
            counter[pair] += 1
    # order by key
    ordered_feat = sorted(counter.iteritems())

    # # Count for each layer how many nodes are activated
    # pCount = Counter(map(lambda x: x[0][0], ordered_feat))
    # pCountS = sorted(pCount.iteritems())

    # # Create probability for each node
    # proba_feat = []
    # for i in ordered_feat:
    #     cur_layer = i[0][0]
    #     elem = (i[0], float(i[1]) / pCountS[cur_layer][1])
    #     print elem
    #     proba_feat.append(elem)

    # Get max histogram
    max_node_occurence = max(counter.values())

    # Create probability for each node
    proba_feat = []
    for i in ordered_feat:
        elem = (i[0], float(i[1]) / max_node_occurence)
        proba_feat.append(elem)

    p = Component(comp_id)
    p.cluster_id = comp_list[0].cluster_id
    p.reconstruct_from_list(proba_feat, tol)

    return p


def filter_noisy_cluster(grouped_components):
    filt_grouped_components = OrderedDict()
    for i, cluster in enumerate(grouped_components):
        if len(cluster) > 1:
            active_nodes = Counter(itertools.chain(*[c.func_ids for c in cluster]))
            # if more than 2 functional ids
            # are in all the nodes of the same cluster
            if len(active_nodes) > 1:
                v = sorted(active_nodes.values(), reverse=True)
                # if the most represented value is at least
                # two times more present than the second most represented one
                if v[0] >= 2 * v[1]:
                    filt_grouped_components[i] = cluster
            else:
                filt_grouped_components[i] = cluster

    return filt_grouped_components


def plot_component_repartition(figsize, components, cluster_count, columns=8):
    plt.figure(1, figsize)
    nn = cluster_count
    rows = math.ceil(float(len(components)) / columns)
    for i in xrange(nn):
        plt.subplot(rows, columns, i+1)
        active_nodes = list(itertools.chain(*[ c.func_ids for c in components[i]]))
        plt.hist(active_nodes, FUNC_ZONES)
        plt.title('Cluster id: ' + str(i))

    plt.tight_layout()
    plt.show()


def plot_component_width_and_height(figsize, components, title='Component'):
    fig = plt.figure(1, figsize)
    ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    ax1.bar(np.arange(len(components)),
            [c.height() for c in components], 1)
    plt.xlabel(title + ' id')
    plt.ylabel('Height')
    plt.title(title + ' heights')

    ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, first plot
    ax2.bar(np.arange(len(components)),
            [c.width() for c in components], 1)
    plt.xlabel(title + ' id')
    plt.ylabel('Width')
    plt.title(title + ' widths')
    plt.tight_layout()
    plt.show()


def plot_component_func_repartition(figsize, components,
                                    columns=4, title='Component'):
    fig = plt.figure(1, figsize)
    rows = math.ceil(float(len(components)) / columns)
    for i, c in enumerate(components):
        # create histo for each pattern
        counter = Counter(c.func_ids)
        # add 0 for mimssing values
        for f in xrange(1, FUNC_ZONES+1):
            if f not in counter:
                counter[f] = 0
        # ordered by func id
        val = OrderedDict(sorted(counter.iteritems(), key=lambda t: t[0]))
        if sum(val.values()) > 0:
            # plot
            ax = fig.add_subplot(rows, columns, i+1)
            ax.bar(np.arange(1, FUNC_ZONES+1), val.values(), 0.8)
            plt.xlabel('Functional id')
            plt.ylabel('#nodes')
            plt.title(title + ' ' + str(i))
            plt.xticks(np.arange(1, FUNC_ZONES+1))

    plt.tight_layout()
    plt.show()


class Component(object):

    def __init__(self, component_id):
        self.id = component_id
        self.g = nx.Graph()
        self.layers = None
        self.input_ids = None
        self.feat = None
        self.ts_group = 0
        self.cluster_id = None
        self.func_ids = None
        self.compfeat = None  # compressed features
        self.subgraphs = None

    def add_node(self, nid, data):
        self.g.add_node(nid, data)

    def add_edge(self, src, tgt, data):
        self.g.add_edge(src, tgt, data)

    def reconstruct_from_array(self, feature, tol=0.0):
        self.g = build_graph_from_activated_layers(feature, tol)
        self.extract_properties()

    def reconstruct_from_list(self, ordered_feat, tol=0.0):
        self.g = build_graph_from_feature_tuples(ordered_feat, tol)
        self.extract_properties()

    def extract_properties(self):
        self.layers = []
        self.feat = []
        self.func_ids = []

        inIds = Counter()
        layer_positions = OrderedDict()
        last_pos = 0

        # iter through nodes to extract properties such as
        # layers, width, height, features, compressed features
        for n, d in self.g.nodes_iter(data=True):
            self.ts_group = d['ts_group_id']
            lp = d['layer_pos']
            input_id = d['input_id']
            self.func_ids.append(int(d['func_id']))

            # needed to compute relative postion (0, 1, 2) of this component
            if lp not in layer_positions:
                layer_positions[lp] = last_pos
                last_pos += 1

            relative_pos = layer_positions[lp]
            self.feat.append((relative_pos, input_id))
            inIds.update({input_id: 1})

        self.layers = sorted(layer_positions.keys())
        self.input_ids = OrderedDict(sorted(
            inIds.iteritems(), key=lambda t: t[0]))

        # Compute probability for each input id
        # self.compfeat = [(k, float(v) / self.width())
        #                  for k, v in self.input_ids.iteritems()]

        self.compfeat = [(k, v) for k, v in self.input_ids.iteritems()]
        # Extract subgraphs
        self.subgraphs = nx.connected_component_subgraphs(self.g)

    def width(self):
        return len(self.layers)

    def height(self):
        """
        Return the count of input_ids (original brain node ids)
        over which the component spans
        """
        return len(self.input_ids)

    def features(self):
        """
        Return list of pairs where tuple[0] is the layer
        number and tuple[1] the position where the node is activated
        """
        return self.feat

    def compressed_features(self):
        """
        Return a list of pair where pair[0] is a tuple (layer, input_id)
        and pair[1] the weight assocatied to this entry
        """
        return self.compfeat

    def node_count(self):
        return self.g.number_of_nodes()

    def edge_count(self):
        return self.g.number_of_edges()

    def layers(self):
        return self.layers

    def patterns(self):
        if not self.subgraphs:
            return None
        patterns = []
        for i, s in enumerate(self.subgraphs):
            p = Pattern(self.id, i)
            p.g = s  # set graph attr
            p.extract_properties()
            patterns.append(p)
        return patterns

    def _draw_graph(self, fig, ax, g):
        pos = nx.get_node_attributes(g, 'pos')
        color = nx.get_node_attributes(g, 'weight').values()
        nx.draw_networkx_edges(g, pos, alpha=0.4)
        nx.draw_networkx_nodes(g, pos,
                               node_size=80,
                               node_color=color,
                               cmap=plt.cm.PuOr,
                               vmin=0.0, vmax=1.0)

        label_start_x = -7
        label_end_x = -2
        for k, p in pos.iteritems():
            lab = str(g.node[k]['input_id']) + ' ' + \
                g.node[k]['name'] + ' ' + \
                str(g.node[k]['func_id'])
            ax.text(label_start_x, p[1], lab, fontsize=14)

        last_layer = self.layers[-1]
        full_width = last_layer + 1
        x_labels = np.arange(full_width)
        x_ticks = np.arange(0, (full_width * X_SPACING), X_SPACING)

        plt.xticks(x_ticks, x_labels, size='large')
        plt.xlim(label_start_x - 1,
                 label_end_x + X_SPACING + last_layer * X_SPACING)

        # last_id = self.input_ids.keys()[-1]
        # plt.ylim(0, last_id * Y_SPACING)
        ax.set_yticklabels([])
        ax.grid(False)

    def draw(self, figid=None, figsize=None):
        if self.g.number_of_nodes() == 0:
            print 'Empty graph for component', self.id
            return None
        subgraph_height = 10
        if not figsize:
            figsize = (16, len(self.subgraphs) * subgraph_height + 1)
        fig = plt.figure(figid, figsize=figsize, dpi=360)
        fig.set_visible(False)
        fig.suptitle('Patterns of component ' +
                     str(self.id), fontsize=14, fontweight='bold')
        ax1 = None
        for i, s in enumerate(self.subgraphs):
            ax = fig.add_subplot(len(self.subgraphs), 1, i)
            if i == 0:
                ax1 = ax
            self._draw_graph(fig, ax, s)
        ax1.set_xlabel('time step')  # set only for last figure
        return fig

    def __str__(self):
        s = 'Component id: ' + str(self.id) + '\n'
        if self.cluster_id:
            s += ' cluster id: ' + str(self.cluster_id) + '\n'
        if len(self.subgraphs) > 0:
            s += ' subgraphs: ' + str(len(self.subgraphs)) + '\n'
        r = ' group: ' + str(self.ts_group) + '\n' \
            ' #nodes: ' + str(self.g.number_of_nodes()) + '\n' \
            ' #edges: ' + str(self.g.number_of_edges()) + '\n' \
            ' #width: ' + str(self.width()) + ' -- ' + \
            ', '.join(map(str, self.layers)) + '\n' \
            ' #height ' + str(self.height()) + '\n'
        return s + r

    def __repr__(self):
        return self.__str__()


class Pattern(Component):

    def __init__(self, component_id, pattern_id):
        super(Pattern, self).__init__(component_id)
        self.pid = pattern_id


class Patient:

    def __init__(self, patient_id):
        self.pid = patient_id
        self.components = []

    def add_component(self, comp):
        self.components.append(comp)

    def aggregate(self, comp_prop):
        return map(lambda x: getattr(x, comp_prop)(), self.components)

    def apply_on_components(self, func, comp_prop):
        # Dynamically call funtion defined by user on each component method
        return func(self.aggregate(comp_prop))

    def mean(self, comp_prop):
        if not self.components:
            return None
        f = lambda x: float(sum(x)) / len(x)
        # calc mean
        m = self.apply_on_components(f, comp_prop)
        # standard dev
        std = math.sqrt(
            f(map(lambda x: (x - m) ** 2, self.aggregate(comp_prop))))
        return m, std

    def component_count(self):
        return len(self.components)

    def node_count(self):
        return self.apply_on_components(sum, 'node_count')

    def edge_count(self):
        return self.apply_on_components(sum, 'edge_count')

    def __str__(self):
        s = 'Patient id: ' + str(self.pid) + '\n' \
            '#components: ' + str(self.component_count()) + '\n' \
            '#nodes: ' + str(self.node_count()) + '\n' \
            '#edges: ' + str(self.edge_count()) + '\n' \
            'width (mean, std): ' + str(self.mean('width')) + '\n' \
            'height (mean, std): ' + str(self.mean('height')) + '\n'
        return s
