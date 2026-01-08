#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, det
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shape3D(nx.Graph):
    def __init__(self, nodes, edges):
        super().__init__()
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        self.sides = self.init_sides()

    def init_sides(self):
        simple_cycles = sorted(nx.simple_cycles(self))
        shortest_simple_cycles = []
        for cycle in simple_cycles:
            v1 = [self.nodes[cycle[1]]["pos"][i] - self.nodes[cycle[0]]["pos"][i] for i in range(3)]
            v2 = [self.nodes[cycle[1]]["pos"][i] - self.nodes[cycle[2]]["pos"][i] for i in range(3)]
            intersect = self.nodes[cycle[1]]["pos"]
            A, B, C, D = 0, 0, 0, 0
            A_matrix = np.array([[v1[1], v1[2]],
                                 [v2[1], v2[2]]])
            if det(A_matrix) != 0:
                b_vector = np.array([[-v1[0]],
                                     [-v2[0]]])
                params = solve(A_matrix, b_vector)
                A = 1
                B = params[0]
                C = params[1]
            else:
                A_matrix = np.array([[v1[0], v1[2]],
                                     [v2[0], v2[2]]])
                if det(A_matrix) != 0:
                    b_vector = np.array([[-v1[1]],
                                         [-v2[1]]])
                    params = solve(A_matrix, b_vector)
                    A = params[0]
                    B = 1
                    C = params[1]
                else:
                    A_matrix = np.array([[v1[0], v1[1]],
                                         [v2[0], v2[1]]])
                    b_vector = np.array([[-v1[2]],
                                         [-v2[2]]])
                    params = solve(A_matrix, b_vector)
                    A = params[0]
                    B = params[1]
                    C = 1
            D = -A * intersect[0] - B * intersect[1] - C * intersect[2]

            def plane(point):
                return A * point[0] + B * point[1] + C * point[2] + D

            for node in cycle:
                if plane(self.nodes[node]["pos"]) != 0:
                    break
            else:
                shortest_simple_cycles.append(cycle)
        return shortest_simple_cycles

    def plot3Dshape(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure(figsize=(8, 6))
        if ax is None:
            ax = fig.add_subplot(projection='3d')
        verts = []
        for side in self.sides:
            verts_i = [self.nodes[side[i]]["pos"] for i in range(len(side))]
            verts.append(verts_i)
        poly = Poly3DCollection(
            verts,
            facecolor="cyan",
            edgecolor="k",
            alpha=0.4
        )
        ax.add_collection3d(poly)
        plt.show()
        return fig, ax
