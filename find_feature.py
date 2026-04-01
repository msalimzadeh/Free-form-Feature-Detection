from config import *
from graph import *
from remove_feature import *
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect free-form feature, extrude it, and intersect with stock removal volume."
    )
    parser.add_argument("--step", required=True, help="Path to finished part STEP file")
    parser.add_argument("--stock", required=True, help="Path to stock STEP file")
    parser.add_argument(
        "--out",
        default="feature_removal.stl",
        help="Output stl file for intersected feature removal volume",
    )
    args = parser.parse_args()

    step_path = args.step
    stock_path = args.stock

    # Load part and stock shapes
    shape, faces_list = read_step_from_user(step_path)  
    stock_shape, _ = read_step_from_user(stock_path)
    
    G = build_face_adjacency(faces_list)    
    attach_face_attributes(G, faces_list)
    attach_edge_angles(G, faces_list)

    picked_face = pick_brep_face(shape, faces_list)
    face_id = faces_list.index(picked_face)
    G.nodes[face_id]["flag"] = True

    feature_faces = grow_region(G, face_id)
    detected_patch_faces = [faces_list[i] for i in feature_faces]

    # Extrude the detected patch to obtain a feature volume (prism)
    removal_prism = extrude_feature_patch(
        detected_patch_faces,
        direction=(0,0,1),   # machining direction
        length=100,          # large enough to reach stock top
    )
    extrusion = "extrusion.step"
    save_shape_to_step(removal_prism, extrusion)

    mesh_step = step_to_stl(step_path)
    mesh_stock = step_to_stl(stock_path)
    mesh_extrusion = step_to_stl(extrusion)
    
    removal_mesh(mesh_extrusion, mesh_stock, args.out)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
