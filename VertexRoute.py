#!/usr/bin/python
# -*- coding: utf-8 -*-

from Shape3D import spaceEnum, Shape3D, Block
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

    for phi in np.linspace(0, np.pi, 10, endpoint=False):
        n = np.sin(phi) * v1 + np.cos(phi) * v2
        D = -np.dot(n, T)

        plane = {
            "A": n[0],
            "B": n[1],
            "C": n[2],
            "D": -np.dot(n, T)
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
            # PLOTOWANIE TESTOWE
            viable_points = np.array(viable_points).reshape(len(viable_points), 3)
            fig, ax = block_object.plot3Dshape()
            ax.scatter(viable_points[:, 0], viable_points[:, 1], viable_points[:, 2],
                        s=120,
                        color='red',
                        edgecolors='black',
                        linewidths=1)
            plt.draw()
            plt.show()
            break

