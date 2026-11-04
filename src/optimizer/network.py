from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox

from .domain import ROUTES, PlanningData, load_planning_data


@dataclass
class TravelNetwork:
    ids: list[str]
    seconds: np.ndarray
    coordinates: list[tuple[float, float]]  # lat, lon; depot at index 0
    provenance: dict
    graph: nx.MultiDiGraph | None = None
    nodes: list[int] | None = None

    def path(self, a: int, b: int):
        if self.graph is None:
            return [self.coordinates[a], self.coordinates[b]]
        try:
            route = nx.shortest_path(self.graph, self.nodes[a], self.nodes[b], weight="travel_time")
        except nx.NetworkXNoPath:
            return [self.coordinates[a], self.coordinates[b]]
        points = []
        for u, v in zip(route, route[1:]):
            edge = min(self.graph[u][v].values(), key=lambda x: x["travel_time"])
            if "geometry" in edge:
                xy = list(edge["geometry"].coords)
                if abs(xy[0][0]-self.graph.nodes[u]["x"]) > abs(xy[-1][0]-self.graph.nodes[u]["x"]):
                    xy.reverse()
                points.extend((float(y), float(x)) for x, y in xy)
            else:
                points.extend((float(self.graph.nodes[n]["y"]), float(self.graph.nodes[n]["x"])) for n in (u, v))
        return points or [self.coordinates[a], self.coordinates[b]]


def configure_osm():
    ROUTES.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = ROUTES / "osm_cache"
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 120


def download_network():
    configure_osm()
    graph_file = ROUTES / "yuseong_drive.graphml"
    boundary_file = ROUTES / "yuseong_boundary.geojson"
    if graph_file.exists():
        return ox.load_graphml(graph_file)
    boundary = ox.geocode_to_gdf("Yuseong-gu, Daejeon, South Korea")
    boundary.to_file(boundary_file, driver="GeoJSON")
    # Include a buffer so short connecting trips can leave/re-enter the district.
    polygon = boundary.to_crs(5179).geometry.buffer(1200).to_crs(4326).iloc[0]
    graph = ox.graph_from_polygon(polygon, network_type="drive", retain_all=True)
    ox.save_graphml(graph, graph_file)
    (ROUTES / "network_source.json").write_text(json.dumps({
        "source": "OpenStreetMap / OSMnx drive graph", "downloaded_unix": time.time(),
        "boundary": "Yuseong-gu + 1200 m buffer", "nodes": len(graph), "edges": graph.number_of_edges(),
    }, ensure_ascii=False, indent=2))
    return graph


def get_depot():
    configure_osm()
    path = ROUTES / "depot.json"
    if path.exists():
        obj = json.loads(path.read_text())
        return float(obj["lat"]), float(obj["lon"])
    coordinates = ox.geocode("대전광역시 유성구 대학로 211")
    path.write_text(json.dumps({"label": "유성구청", "address": "대전광역시 유성구 대학로 211",
        "lat": coordinates[0], "lon": coordinates[1], "method": "OSM Nominatim address"}, ensure_ascii=False, indent=2))
    return coordinates


def build_matrix(data: PlanningData, speed_kmh=25., *, mock=False, load_geometry=True):
    if not 0 < speed_kmh <= 100:
        raise ValueError("도시부 평균속도는 0 초과 100 km/h 이하로 지정하세요.")
    ROUTES.mkdir(parents=True, exist_ok=True)
    depot = (36.3622, 127.3561) if mock else get_depot()
    coordinates = [depot]+list(zip(data.spots.lat.astype(float), data.spots.lon.astype(float)))
    ids = ["DEPOT"]+data.spots.spot_id.astype(str).tolist()
    identity = json.dumps({"ids": ids, "coordinates": coordinates, "speed": speed_kmh, "mock": mock})
    # This hash binds the matrix to an exact ordered coordinate/version snapshot.
    key = hashlib.sha256(identity.encode()).hexdigest()[:18]
    cached = ROUTES / f"{'mock_' if mock else ''}matrix_{key}.npz"
    if mock:
        c = np.asarray(coordinates)
        xy = c * [111_000, 89_500]
        matrix = np.ceil(np.linalg.norm(xy[:, None]-xy[None, :], axis=2)/(speed_kmh/3.6)).astype(np.int64)
        return TravelNetwork(ids, matrix, coordinates, {"source": "MOCK straight-line; not a road evaluation", "speed_kmh": speed_kmh})
    graph = download_network()
    # A strongly connected drive component guarantees directed trips and returns.
    component = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    for _, _, edge in graph.edges(data=True):
        edge["travel_time"] = float(edge["length"])/(speed_kmh/3.6)
    nodes, distances = ox.distance.nearest_nodes(graph, [c[1] for c in coordinates], [c[0] for c in coordinates], return_dist=True)
    nodes = list(map(int, nodes))
    if max(distances) > 600:
        raise ValueError(f"도로망과 600 m 이상 떨어진 좌표가 있습니다 (최대 {max(distances):.0f} m). 지점 좌표를 대조하세요.")
    if cached.exists():
        matrix = np.load(cached)["seconds"]
    else:
        matrix = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
        lengths_by_node = {}
        for i, node in enumerate(nodes):
            if node not in lengths_by_node:
                lengths_by_node[node] = nx.single_source_dijkstra_path_length(graph, node, weight="travel_time")
            lengths = lengths_by_node[node]
            matrix[i] = [int(np.ceil(lengths[target])) for target in nodes]
        np.savez_compressed(cached, seconds=matrix)
    provenance = {"source": "OSM directed drive shortest paths", "speed_kmh": speed_kmh,
        "matrix_file": str(cached.relative_to(ROUTES)), "graph_nodes": len(graph),
        "snap_max_m": float(max(distances)), "offroad_snap_time_included": False,
        "depot": "유성구청, 대전 유성구 대학로 211"}
    return TravelNetwork(ids, matrix, coordinates, provenance, graph if load_geometry else None, nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        get_depot()
        graph = download_network()
        print(json.dumps({"graph_nodes": len(graph), "graph_edges": graph.number_of_edges()}))
    else:
        data = load_planning_data(ROUTES if args.mock else None, mock=args.mock)
        network = build_matrix(data, mock=args.mock)
        print(json.dumps(network.provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
