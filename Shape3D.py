#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, det
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from enum import Enum

class spaceEnum(Enum):
    SPACE_LIMITS = 0
    OBJECTS = 1

class planeEnum(Enum):
    CYCLE = 0
    PLANE = 1


class Shape3D(nx.Graph):
    def __init__(self, nodes, edges):
        super().__init__()
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        self.sides = self.init_sides()
        edges_with_lines = self.init_edges()
        self.remove_edges_from(edges)
        self.add_edges_from(edges_with_lines)

    def init_edges(self):
        edges_with_lines = []
        for edge in self.edges:
            planes = []
            for cycle, plane in self.sides:
                if edge[0] in cycle and edge[1] in cycle:
                    planes.append(plane)
            line = planes
            edges_with_lines.append((edge[0], edge[1], {"line_eq": line}))

        return edges_with_lines

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
                A = 1.0
                B = float(params[0, 0])
                C = float(params[1, 0])
            else:
                A_matrix = np.array([[v1[0], v1[2]],
                                     [v2[0], v2[2]]])
                if det(A_matrix) != 0:
                    b_vector = np.array([[-v1[1]],
                                         [-v2[1]]])
                    params = solve(A_matrix, b_vector)
                    A = float(params[0, 0])
                    B = 1.0
                    C = float(params[1, 0])
                else:
                    A_matrix = np.array([[v1[0], v1[1]],
                                         [v2[0], v2[1]]])
                    b_vector = np.array([[-v1[2]],
                                         [-v2[2]]])
                    params = solve(A_matrix, b_vector)
                    A = float(params[0, 0])
                    B = float(params[1, 0])
                    C = 1.0
            D = -A * intersect[0] - B * intersect[1] - C * intersect[2]

            plane = {"A": A,
                     "B": B,
                     "C": C,
                     "D": D}
            for node in cycle:
                point = self.nodes[node]["pos"]
                if A * point[0] + B * point[1] + C * point[2] + D != 0:
                    break
            else:
                shortest_simple_cycles.append((cycle, plane))
        return shortest_simple_cycles

    def plot3Dshape(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure(figsize=(8, 6))
        if ax is None:
            ax = fig.add_subplot(projection='3d')

        verts = []

        for side, _ in self.sides:
            verts_i = [self.nodes[i]["pos"] for i in side]
            verts.append(verts_i)

        poly = Poly3DCollection(
            verts,
            facecolor="cyan",
            edgecolor="k",
            alpha=0.1
        )
        ax.add_collection3d(poly)

        return fig, ax


class Block(Shape3D):
    def __init__(self, center, length, width, height):
        n0_pos = [center[0] - length/2, center[1] - width/2, center[2] - height/2]
        n1_pos = list(n0_pos)
        n1_pos[0] += length
        n2_pos = list(n0_pos)
        n2_pos[1] += width
        n3_pos = list(n0_pos)
        n3_pos[2] += height
        n4_pos = list(n1_pos)
        n4_pos[1] += width
        n5_pos = list(n1_pos)
        n5_pos[2] += height
        n6_pos = list(n2_pos)
        n6_pos[2] += height
        n7_pos = list(n4_pos)
        n7_pos[2] += height
        node_pos = [n0_pos, n1_pos, n2_pos, n3_pos,
                    n4_pos, n5_pos, n6_pos, n7_pos]
        nodes = []
        for i in range(8):
            nodes.append((i, {"pos": node_pos[i]}))
        edges = [(0, 1),
                 (0, 2),
                 (0, 3),
                 (1, 4),
                 (1, 5),
                 (2, 4),
                 (2, 6),
                 (3, 5),
                 (3, 6),
                 (4, 7),
                 (5, 7),
                 (6, 7)]
        super().__init__(nodes, edges)

    def if_in(self, point):
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        P1 = np.array(self.nodes[0]["pos"])
        P2 = np.array(self.nodes[7]["pos"])
        return (point > P1).all() and (point < P2).all()

