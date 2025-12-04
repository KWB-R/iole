import os
from itertools import cycle
from dataclasses import dataclass
from typing import Literal, List, Dict
from more_itertools import windowed

import oopnet as on
import networkx as nx
import matplotlib.pyplot as plt


def get_number_of_sensors(network_length_m: float, m_per_sensor: float, _min: int = 1):
    return int(max(network_length_m // m_per_sensor, _min))


def distribute_integer(n: int, weightings: dict[str:float], _min=1):
    _check = _min * len(weightings.keys())
    assert _check <= n, "No solution possible for given parameters."

    _dictsum = lambda x: sum(list(x.values()))

    per_key = {k: p * n for k, p in weightings.items()}
    result = {k: round(v) for k, v in per_key.items()}

    _fixed = []
    for k in result.keys():
        if result[k] < _min:
            result[k] = _min
            _fixed.append(k)

    if all(k in _fixed for k in result.keys()):
        return result

    overallocation = _dictsum(result) - n

    if overallocation == 0:
        return result

    # round robin for parameters that werent fixed to get rid of the overallocation

    if overallocation < 0:
        eligible_keys = iter(
            [
                k
                for k in dict(
                    sorted(weightings.items(), key=lambda x: x[1], reverse=True)
                ).keys()
                if k not in _fixed
            ]
        )
        for _ in range(abs(overallocation)):
            result[next(eligible_keys)] += 1

    if overallocation > 0:
        eligible_keys = cycle(
            [
                k
                for k in dict(
                    sorted(weightings.items(), key=lambda x: x[1], reverse=True)
                ).keys()
                if k not in _fixed
            ]
        )
        while overallocation > 0:
            k = next(eligible_keys)
            if result[k] >= _min + 1:
                result[k] -= 1
                overallocation -= 1

    return result


def get_edge_by_attr(G: nx.Graph, attr_key: str, *attr_values) -> List:
    result = [e for e in G.edges(data=True) if e[2].get(attr_key) in attr_values]
    return result


def get_node_by_attr(G: nx.Graph, attr_key: str, *attr_values) -> Dict:
    result = {
        n[0]: n[1] for n in G.nodes(data=True) if n[1].get(attr_key) in attr_values
    }
    return result


@dataclass
class SensorPlacement:

    nw_path: os.PathLike
    reservoir_ids: List[str]

    def __post_init__(self):
        self.selected_nodes = None

    def calculate(self, method: Literal["dummy_pipes", "0_weighted"], n_sensors: int):
        self.selected_nodes = []

        match method:
            case "dummy_pipes":
                _method = 1
            case "0_weighted":
                _method = 2
            case _:
                raise Exception()

        G = self._get_prepared_graph(method=_method)

        for k in range(0, n_sensors):

            print(f"Calculating sensor {k+1}/{n_sensors}")

            pathes = dict.fromkeys(get_node_by_attr(G, "is_candidate", True))

            # Calculate the distance (shortest path length) between the source and the candidate node.
            for node in pathes.keys():
                distance, path = nx.single_source_dijkstra(
                    G, source="superSource", target=node, weight="weight"
                )
                pathes[node] = dict(
                    distance=distance,
                    path_nodes=path,
                    path_edges=[w for w in windowed(path, n=2)],
                )

            # select candidate
            node_name, node_values = max(
                pathes.items(), key=lambda item: item[1]["distance"]
            )

            print(f"Selected node: {node_name}")

            G.nodes[node_name]["is_selected"] = True

            self.selected_nodes.append(node_name)

            match G.graph["method"]:
                case 1:
                    # Select the node with the max shortest path lenght from the source and add a link with weight = 0 (between the source and the selected node)
                    G.add_edge("superSource", node_name)
                    G["superSource"][node_name]["weight"] = 0

                case 2:
                    for edge in node_values["path_edges"]:
                        G[edge[0]][edge[1]]["weight"] = 0

        return G

    def _get_prepared_graph(self, method: Literal[1, 2]) -> nx.DiGraph:

        # Prepare network
        nw = on.Network.read(self.nw_path)
        nw = self._add_super_source(nw)
        nw = self._drop_closed_pipes_valves(nw)

        # Nodes
        _coords = on.get_coordinates(nw)
        xlist = _coords.iloc[:, 0]
        ylist = _coords.iloc[:, 1]
        node_names = on.get_node_ids(nw)
        node_coords = {n: (x, y) for n, x, y in zip(node_names, xlist, ylist)}

        # Edges
        edges = {(l.startnode.id, l.endnode.id): l.id for l in on.get_links(nw)}

        # Weights
        weights = {}

        for pipe in on.get_pipes(nw):
            weights[(pipe.startnode.id, pipe.endnode.id)] = pipe.length

        for valve in on.get_valves(nw):
            weights[(valve.startnode.id, valve.endnode.id)] = 1

        # Add to Graph
        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        G.add_edges_from(list(edges.keys()))
        nx.set_edge_attributes(G, edges, name="id")

        # Set weights
        try:
            nx.set_edge_attributes(G, weights, name="weight")
        except Exception as e:
            print(e)

        # Set position
        try:
            nx.set_node_attributes(G, node_coords, name="pos")
        except Exception as e:
            print(e)

        # Set metadata
        G.graph["method"] = method

        # Find dead ends
        dead_end = []
        for node in G.nodes:
            summ_degrees = G.in_degree()[node] + G.out_degree()[node]
            if summ_degrees == 1:
                dead_end.append(node)

        # Convert to undirected:
        G = G.to_undirected()

        # Create a list of candidate node. In this list the death end points are excluded.
        candidate_nodes = {
            n: True if n not in dead_end and "node_n" not in n else False
            for n in G.nodes
        }
        selected_nodes = {n: False for n in G.nodes}

        nx.set_node_attributes(G, candidate_nodes, name="is_candidate")
        nx.set_node_attributes(G, selected_nodes, name="is_selected")

        return G

    def _add_super_source(self, nw: on.Network) -> on.Network:
        reservoir_nodes = [
            on.get_node(nw, id=reservoir_id) for reservoir_id in self.reservoir_ids
        ]
        x_coord = sum([node.xcoordinate for node in reservoir_nodes]) / len(
            self.reservoir_ids
        )
        y_coord = sum([node.ycoordinate for node in reservoir_nodes]) / len(
            self.reservoir_ids
        )

        on.add_reservoir(
            nw, on.Reservoir(id="superSource", xcoordinate=x_coord, ycoordinate=y_coord)
        )

        for i, reservoir in enumerate(self.reservoir_ids):
            on.add_pipe(
                nw,
                on.Pipe(
                    id=f"dummy_pipe_{i}",
                    length=1,
                    startnode=on.Junction(id="superSource"),
                    endnode=on.Junction(id=reservoir),
                ),
            )

        return nw

    def _drop_closed_pipes_valves(self, nw):
        for pipe in on.get_pipes(nw):
            if pipe.status != "OPEN":
                on.remove_pipe(nw, pipe.id)
        for valve in on.get_valves(nw):
            if valve.status != "OPEN":
                on.remove_valve(nw, valve.id)

        return nw

    @staticmethod
    def plot_sensor_placement_graph(G: nx.Graph):
        fig, ax = plt.subplots(1, 1, figsize=(25, 16), dpi=300)

        pos = nx.get_node_attributes(G, "pos")
        weights = nx.get_edge_attributes(G, "weight")

        node_colors = [
            "r" if attrs["is_selected"] else "k" for n, attrs in G.nodes(data=True)
        ]
        node_sizes = [
            30 if attrs["is_selected"] or (n == "superSource") else 0.25
            for n, attrs in G.nodes(data=True)
        ]

        edge_transparencies = [1 if w == 0 else 0.3 for w in weights.values()]
        edge_colors = ["g" if w == 0 else "k" for w in weights.values()]

        nx.draw(
            G.to_undirected(),
            pos=pos,
            ax=ax,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            alpha=edge_transparencies,
        )

        plt.show()


if __name__ == "__main__":
    test_nw = r"C:\Users\jkoslo\Documents\iOLE (lok)\programming\iole_kwb\iole\data\networks\test_slice.inp"

    spl = SensorPlacement(test_nw, reservoir_ids=["157087-A", "KT19807", "KT19235"])

    G = spl.calculate(method="dummy_pipes", n_sensors=2)

    print(spl.selected_nodes)
