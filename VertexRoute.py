#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx

from Shape3D import spaceEnum
import numpy as np
import matplotlib.pyplot as plt

def find_vertex_route(space, starting_point, ending_point):
    starting_point = np.array(starting_point)
    ending_point = np.array(ending_point)
    u = ending_point - starting_point
    u = u / np.linalg.norm(u)
    x0 = starting_point
    t = -sum(x0 * u) / sum(u * u)
    T = x0 + u * t

    if abs(u[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    v1 = np.cross(u, a)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(u, v1)
    best_path = {"path": None, "length": np.inf}
    for phi in np.linspace(0, np.pi, 180, endpoint=False):
        n = np.sin(phi) * v1 + np.cos(phi) * v2
        D = -np.dot(n, T)

        plane = {
            "A": n[0],
            "B": n[1],
            "C": n[2],
            "D": D
        }

        viable_points = []
        for block_object in space[spaceEnum.OBJECTS]:
            for edge in block_object.edges:
                planes = block_object.edges[edge]["line_eq"]
                eq_planes = planes + [plane]
                N = len(eq_planes)
                A = np.zeros([N] * 2)
                b = np.zeros([N, 1])
                A[:, :] = [[eq_planes[i][letter] for letter in ["A", "B", "C"]] for i in range(N)]
                b[:, 0] = [-eq_planes[i]["D"] for i in range(N)]
                if np.linalg.det(A) == 0:
                    continue
                x = np.linalg.solve(A, b)
                x = x.flatten()
                P1 = np.array(block_object.nodes[edge[0]]["pos"])
                P2 = np.array(block_object.nodes[edge[1]]["pos"])
                limits_down = np.min(np.array([P1, P2]), axis=0)
                limits_up = np.max(np.array([P1, P2]), axis=0)
                if (x < limits_down).any() or (x > limits_up).any():
                    continue
                viable_points.append(x)

        viable_points = [tuple(p) for p in viable_points]
        starting_point = tuple(starting_point)
        ending_point = tuple(ending_point)
        viable_points.append(starting_point)
        viable_points.append(ending_point)

        G = nx.Graph()
        G.add_nodes_from(viable_points)

        points_to_check = [starting_point, ending_point]

        while len(points_to_check) != 0:
            point1 = points_to_check.pop()
            for point2 in G.nodes:
                if np.all(point1 == point2) or (point1, point2) in G.edges:
                    continue
                point_viable = True
                u = np.array(point2) - np.array(point1)
                x0 = point1
                t_vector = np.linspace(0, 1, 100)
                for t in t_vector:
                    x = x0 + t * u
                    for block_object in space[spaceEnum.OBJECTS]:
                        if block_object.if_in(x):
                            point_viable = False
                            break
                if point_viable:
                    G.add_edge(point1, point2, weight=np.linalg.norm(np.array(point2) - np.array(point1)))
                    points_to_check.append(point2)
        shortest_path = nx.shortest_path(G, source=starting_point, target=ending_point, weight="weight")
        edges_in_path = list(zip(shortest_path[:-1], shortest_path[1:]))
        path = {"path": shortest_path, "length": sum([G.edges[edge]["weight"] for edge in edges_in_path])}
        if path["length"] < best_path["length"]:
            best_path = path
    # PLOTOWANIE TESTOWE
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    for block_object in space[spaceEnum.OBJECTS]:
        block_object.plot3Dshape(fig, ax)
    path_points = np.array(best_path["path"])
    ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2],
                s=120,
                color='red',
                edgecolors='black',
                linewidths=1)
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2])
    plt.draw()
    plt.show()
    return best_path

