"""
B-rep face adjacency graph from a STEP file.
Run: python graph.py --step /path/to/model.step

"""

from __future__ import annotations
import argparse
import sys
from typing import Any
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
)
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_Orientation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

SURFACE_TYPE_MAP = {
    GeomAbs_Plane: "Plane",
    GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone",
    GeomAbs_Sphere: "Sphere",
    GeomAbs_Torus: "Torus",
    GeomAbs_BezierSurface: "Bezier",
    GeomAbs_BSplineSurface: "BSpline",
    GeomAbs_SurfaceOfRevolution: "Revolution",
    GeomAbs_SurfaceOfExtrusion: "Extrusion",
    GeomAbs_OffsetSurface: "Offset",
    GeomAbs_OtherSurface: "Other",
}

SURFACE_TYPE_COLOR = {
    "Plane": "#4CAF50",
    "Cylinder": "#2196F3",
    "Cone": "#FF9800",
    "Sphere": "#9C27B0",
    "Torus": "#E91E63",
    "Bezier": "#00BCD4",
    "BSpline": "#009688",
    "Revolution": "#795548",
    "Extrusion": "#607D8B",
    "Offset": "#CDDC39",
    "Other": "#9E9E9E",
}

def load_step(path: str):
    reader = STEPControl_Reader()
    reader.ReadFile(path)
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape

def shape_hash(shape: TopoDS_Shape) -> int:
    try:
        return shape.HashCode(2147483647)
    except AttributeError:
        return hash(shape.TShape())


def _edge_key(edge: TopoDS_Edge):
    verts = []
    expv = TopExp_Explorer(edge, TopAbs_VERTEX)
    while expv.More():
        v = TopoDS.Vertex_s(expv.Current())
        p = BRep_Tool.Pnt_s(v)
        verts.append(
            (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
        )
        expv.Next()
    verts.sort()
    return tuple(verts)

# Face ID mapping
def iter_faces(shape: TopoDS_Shape):
    faces_list: list[Any] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        faces_list.append(face)
        explorer.Next()
    # Optional lookup map if needed elsewhere, does not affect adjacency
    face_key_to_id: dict[int, int] = {
        shape_hash(face): idx for idx, face in enumerate(faces_list)
    }
    return faces_list, face_key_to_id

def _surface_type_for_face(face: TopoDS_Face):
    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()
    return SURFACE_TYPE_MAP.get(stype, f"Other({stype})")

def build_face_adjacency(faces_list):
    # edge_key -> list of face_ids incident to that edge
    edge_to_faces: dict[int, list[int]] = {}

    for face_id, face in enumerate(faces_list):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            # Use geometric, orientation-independent key so the same edge on
            # different faces is always grouped together.
            key = _edge_key(edge)
            if key not in edge_to_faces:
                edge_to_faces[key] = []
            if face_id not in edge_to_faces[key]:
                edge_to_faces[key].append(face_id)
            exp.Next()

    G = nx.Graph()
    for fid in range(len(faces_list)):
        G.add_node(fid, 
                surface_type = _surface_type_for_face(faces_list[fid]), 
                flag = False)

    # (i, j) -> shared_count
    pair_data: dict[tuple[int, int], int] = {}
    for _ek, fids in edge_to_faces.items():
        for ii in range(len(fids)):
            for jj in range(ii + 1, len(fids)):
                i, j = fids[ii], fids[jj]
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key not in pair_data:
                    pair_data[key] = 0
                pair_data[key] = pair_data[key] + 1

    for (i, j), count in pair_data.items():
        G.add_edge(i, j, shared_edge_count=count)

    return G

# We need to also add normal to compare the angle between faces
def compute_face_attributes(face) :
    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()
    surface_type = SURFACE_TYPE_MAP.get(stype, f"Other({stype})")

    gprop = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gprop)
    area = gprop.Mass()
    centre = gprop.CentreOfMass()
    centroid = (centre.X(), centre.Y(), centre.Z())

    return {
        "surface_type": surface_type,
        "area": area,
        "centroid": centroid,
    }


def attach_face_attributes(G, faces_list):
    for face_id, face in enumerate(faces_list):
        if not G.has_node(face_id):
            continue
        attrs = compute_face_attributes(face)
        for k, v in attrs.items():
            G.nodes[face_id][k] = v

def plot_graph(G, title = "Face adjacency graph"):
    pos = nx.spring_layout(G, seed=42, k=1.5)
    node_colors = [
        SURFACE_TYPE_COLOR.get(G.nodes[n].get("surface_type", "Other"), "#9E9E9E")
        for n in G.nodes()
    ]
    labels = {
        # n: f"{n}:{G.nodes[n].get('surface_type', '?')}"
        n: f"{n}"
        for n in G.nodes()
    }

    edges = list(G.edges())
    if not edges:
        widths = []
    else:
        # Use a constant width now that edge length is not tracked
        widths = [2.0 for _ in edges]

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        labels=labels,
        with_labels=True,
        font_size=7,
        edge_color="gray",
        width=widths,
        node_size=400,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
# Mesh Visualization 
def tessellate_shape(shape, linear_deflection = 0.001, angular_deflection = 0.5):
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()

def visualize_3d(shape, faces_list, face_id_map, mesh_deflection = 0.001, angular_deflection = 0.5):

    tessellate_shape(shape, mesh_deflection, angular_deflection)

    all_verts = []
    all_cells = []
    all_face_ids = []
    offset = 0

    for face_id, face in enumerate(faces_list):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        reverse = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED

        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            all_verts.append([p.X(), p.Y(), p.Z()])

        for t in tri.Triangles():
            i1, i2, i3 = t.Value(1), t.Value(2), t.Value(3)
            a, b, c = offset + i1 - 1, offset + i2 - 1, offset + i3 - 1
            if reverse:
                all_cells.extend([3, a, c, b])
            else:
                all_cells.extend([3, a, b, c])
            all_face_ids.append(face_id)

        offset += tri.NbNodes()

    verts = np.array(all_verts, dtype=float)
    cells = np.array(all_cells, dtype=np.int64)
    face_ids = np.array(all_face_ids, dtype=np.int32)

    mesh = pv.PolyData(verts, cells)
    mesh.cell_data["face_id"] = face_ids

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="face_id", show_edges=False, cmap="tab20", show_scalar_bar=True, scalar_bar_args={"title": "face_id"})
    plotter.add_text("Click a face to print face_id and attributes", position="upper_left", font_size=10)

    def on_pick(point):
        cell_id = mesh.find_closest_cell(point)
        fid = int(mesh.cell_data["face_id"][cell_id])
        face = faces_list[fid]
        attrs = compute_face_attributes(face)
        print(f"\nPicked face_id={fid}  surface_type={attrs['surface_type']}  area={attrs['area']:.6f}  centroid={attrs['centroid']}")

    plotter.enable_surface_point_picking(callback=on_pick, show_message=False)
    plotter.show()
def graph_print(G: nx.Graph) -> None:
    for node in G.nodes():
        print (f"surface_type={G.nodes[node]['surface_type']} , flag={G.nodes[node]['flag']}")
        # print (f"{node}: {[n for n in G.neighbors(node)]}, surface_type={G.nodes[node]['surface_type']} , flag={G.nodes[node]['flag']}")
def print_face_table(G: nx.Graph) -> None:
    """Print table: face_id -> surface_type -> area."""
    print("\nface_id | surface_type | area")
    print("-" * 45)
    for n in sorted(G.nodes()):
        st = G.nodes[n].get("surface_type", "?")
        area = G.nodes[n].get("area", 0.0)
        print(f"  {n:4d}  | {st:12s} | {area:.6f}")


def get_plane_face_ids(G) -> list[int]:
    """Return all face IDs (node IDs) whose surface_type is Plane."""
    return [n for n in G.nodes() if G.nodes[n].get("surface_type") == "Plane"]


def main() -> int:
    visualize = True
    parser = argparse.ArgumentParser(description="Build face adjacency graph from a STEP file.")
    parser.add_argument("--step", required=True, help="Path to STEP file")
    args = parser.parse_args()

    step_path = args.step
    shape = load_step(step_path)
    faces_list, face_id_map = iter_faces(shape)
    G = build_face_adjacency(faces_list)    
    attach_face_attributes(G, faces_list)
    print(G)
    print_face_table(G)
    # plot_graph(G, title=f"Face adjacency: {step_path}")
    # visualize_3d(shape, faces_list, face_id_map, mesh_deflection=0.001)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


