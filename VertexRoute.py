#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx

from Shape3D import spaceEnum
import numpy as np
import matplotlib.pyplot as plt

def find_vertex_route(space, starting_point, ending_point, combine_planes = False, number_of_planes = 180):
    space_block = space[spaceEnum.SPACE_LIMITS]
    starting_point = np.array(starting_point)
    ending_point = np.array(ending_point)
    if not space_block.if_in(starting_point) or not space_block.if_in(ending_point):
        print("Punkt startowy lub końcowy poza obszarem dopuszczalnym")
        return None
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
    if combine_planes:
        combined_viable_points = []
        for phi in np.linspace(0, np.pi, number_of_planes, endpoint=False):
            plane = generate_plane(phi, v1, v2, T)

            viable_points = get_cross_points(space, plane)

            combined_viable_points.extend(viable_points)

        combined_viable_points = unique_points(combined_viable_points)

        starting_point = tuple(starting_point)
        ending_point = tuple(ending_point)

        G = initialize_graph(space, combined_viable_points, starting_point, ending_point)

        path = get_shortest_path(G, starting_point, ending_point)

        if path is not None:
            best_path = path

    else:
        for phi in np.linspace(0, np.pi, number_of_planes, endpoint=False):

            plane = generate_plane(phi, v1, v2, T)

            viable_points = get_cross_points(space, plane)

            starting_point = tuple(starting_point)
            ending_point = tuple(ending_point)

            G = initialize_graph(space, viable_points, starting_point, ending_point)

            path = get_shortest_path(G, starting_point, ending_point)

            if path is not None and path["length"] < best_path["length"]:
                best_path = path

    if best_path == {"path": None, "length": np.inf}:
        print("Nie znaleziono ścieżki")
        return None

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
    P1 = space_block.nodes[0]["pos"]
    P2 = space_block.nodes[7]["pos"]
    ax.set_xlim(P1[0], P2[0])
    ax.set_ylim(P1[1], P2[1])
    ax.set_zlim(P1[2], P2[2])

    ax.set_box_aspect([
        P2[0] - P1[0],
        P2[1] - P1[1],
        P2[2] - P1[2]
    ])
    plt.show()
    return best_path

def generate_plane(phi, v1, v2, T):
    n = np.sin(phi) * v1 + np.cos(phi) * v2
    D = -np.dot(n, T)

    plane = {
        "A": n[0],
        "B": n[1],
        "C": n[2],
        "D": D
    }
    return plane

def get_cross_points(space, plane):
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
            if abs(np.linalg.det(A)) < 1e-8:
                continue
            x = np.linalg.solve(A, b)
            x = x.flatten()
            P1 = np.array(block_object.nodes[edge[0]]["pos"])
            P2 = np.array(block_object.nodes[edge[1]]["pos"])
            limits_down = np.min(np.array([P1, P2]), axis=0)
            limits_up = np.max(np.array([P1, P2]), axis=0)
            if (x < limits_down).any() or (x > limits_up).any() or not space[spaceEnum.SPACE_LIMITS].if_in(x):
                continue
            viable_points.append(x)
    return viable_points

def initialize_graph(space, viable_points, starting_point, ending_point):
    viable_points = [tuple(p) for p in viable_points]
    viable_points.append(starting_point)
    viable_points.append(ending_point)

    G = nx.Graph()
    G.add_nodes_from(viable_points)
    fill_graph_with_edges(G, space, starting_point, ending_point)
    return G

def fill_graph_with_edges(G, space, starting_point, ending_point):
    points_to_check = [starting_point, ending_point]

    while len(points_to_check) != 0:
        point1 = points_to_check.pop()
        for point2 in G.nodes:
            if np.all(point1 == point2) or (point1, point2) in G.edges:
                continue
            samples = int(np.linalg.norm(np.array(point2) - np.array(point1)) * 5)
            samples = max(samples, 50)
            if is_visible(point1, point2, space, samples):
                G.add_edge(point1, point2, weight=np.linalg.norm(np.array(point2) - np.array(point1)))
                points_to_check.append(point2)

def get_shortest_path(G, starting_point, ending_point):
    try:
        shortest_path = nx.shortest_path(G, source=starting_point, target=ending_point, weight="weight")
        edges_in_path = list(zip(shortest_path[:-1], shortest_path[1:]))
        path = {"path": shortest_path, "length": sum([G.edges[edge]["weight"] for edge in edges_in_path])}
        return path
    except nx.NetworkXNoPath:
        return None

def unique_points(points, tol=1e-6):
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in unique):
            unique.append(p)
    return unique

def is_visible(p1, p2, space, samples=100):
    u = np.array(p2) - np.array(p1)
    for t in np.linspace(0, 1, samples):
        x = np.array(p1) + t * u
        for block in space[spaceEnum.OBJECTS]:
            if block.if_in(x):
                return False
    return True
