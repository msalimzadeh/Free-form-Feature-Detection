import math
from collections import deque
from typing import Set

import networkx as nx


def grow_region(G: nx.Graph, seed_face_id: int, angle_threshold: float = 40.0) -> Set[int]:
    def is_plane(node_id: int) -> bool:
        return G.nodes[node_id].get("surface_type") == "Plane"
    region: Set[int] = set()
    queue: deque[int] = deque([seed_face_id])
    while queue:
        current = queue.popleft()
        if current in region:
            continue
        if is_plane(current):
            neighbors = list(G.neighbors(current))
            if any(is_plane(nbr) for nbr in neighbors) or not neighbors:
                continue
        region.add(current)
        G.nodes[current]["flag"] = True
        for nbr in G.neighbors(current):
            if nbr in region:
                continue
            angle = G.edges[current, nbr].get("angle_deg", 0.0)
            if math.isnan(angle) or angle > angle_threshold:
                continue
            queue.append(nbr)
    return region
