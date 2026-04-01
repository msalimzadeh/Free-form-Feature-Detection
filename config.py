import sys
import argparse
from pathlib import Path
from typing import Optional

import pyvista as pv
import numpy as np

from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location


def pick_face_on_mesh(mesh: pv.PolyData):
    picked_points = []
    picked_cells = []

    def _on_pick(point):
        picked_points.append(point)
        cid = mesh.find_closest_cell(point)
        picked_cells.append(cid)

    plotter = pv.Plotter()
    base_color = np.array([200, 200, 200], dtype=np.uint8)
    n_cells = mesh.n_cells
    colors = np.tile(base_color, (n_cells, 1))
    mesh.cell_data["colors"] = colors

    plotter.add_mesh(mesh, scalars="colors", rgb=True)
    plotter.enable_surface_point_picking(callback=_on_pick)
    plotter.show()

    cell_id = int(picked_cells[-1])
    point = np.array(picked_points[-1], dtype=float)

    if "face_id" in mesh.cell_data:
        face_id = int(mesh.cell_data["face_id"][cell_id])
    else:
        face_id = cell_id

    return face_id, cell_id, point

def load_step(path: str):
    reader = STEPControl_Reader()
    reader.ReadFile(path)
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape

def read_step_from_user(step_path):

    shape =load_step(step_path)
    # Collect faces
    faces_list: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        faces_list.append(face)
        exp.Next()
    print(f"Loaded STEP: {step_path}  ({len(faces_list)} faces)")
    return shape, faces_list


def _build_face_mesh(shape, faces_list, mesh_deflection = 0.001, angular_deflection = 0.5):
    """
    Tessellate the shape and build a PyVista mesh with cell_data['face_id']
    so that each triangle knows which B-rep face it came from.
    """
    mesh = BRepMesh_IncrementalMesh(shape, mesh_deflection, False, angular_deflection, True)
    mesh.Perform()

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
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            all_verts.append([p.X(), p.Y(), p.Z()])

        for t in tri.Triangles():
            i1, i2, i3 = t.Value(1), t.Value(2), t.Value(3)
            a, b, c = offset + i1 - 1, offset + i2 - 1, offset + i3 - 1
            all_cells.extend([3, a, b, c])
            all_face_ids.append(face_id)

        offset += tri.NbNodes()

    verts = np.array(all_verts, dtype=float)
    cells = np.array(all_cells, dtype=np.int64)
    face_ids = np.array(all_face_ids, dtype=np.int32)

    poly = pv.PolyData(verts, cells)
    poly.cell_data["face_id"] = face_ids
    return poly

# Volume removal visualization
def build_mesh_for_shape(shape, mesh_deflection= 0.001, angular_deflection = 0.5):
    # if we have faces list we can easily remove this function
    faces_list: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        faces_list.append(face)
        exp.Next()

    return _build_face_mesh(
        shape, faces_list,
        mesh_deflection=mesh_deflection,
        angular_deflection=angular_deflection,
    )


def pick_brep_face( shape, faces_list, mesh_deflection = 0.001):
    mesh = _build_face_mesh(shape, faces_list, mesh_deflection=mesh_deflection)
    face_id, cell_id, _ = pick_face_on_mesh(mesh)
    return faces_list[face_id]


def visualize_faces_on_mesh(shape, faces_list, selected_face_ids, mesh_deflection = 0.001):
    mesh = _build_face_mesh(shape, faces_list, mesh_deflection=mesh_deflection)
    selected_ids_set = set(int(i) for i in selected_face_ids)
    face_ids = mesh.cell_data.get("face_id")
    n_cells = mesh.n_cells
    base_color = np.array([200, 200, 200], dtype=np.uint8)
    highlight_color = np.array([255, 0, 0], dtype=np.uint8)
    colors = np.tile(base_color, (n_cells, 1))
    mask = np.isin(face_ids, list(selected_ids_set))
    colors[mask] = highlight_color
    mesh.cell_data["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="colors", rgb=True)
    plotter.show()


def save_shape_to_step(shape: TopoDS_Shape, path: str):

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)


def step_to_stl(
    step_path: str,
    stl_path: Optional[str] = None,
    mesh_deflection: float = 0.001,
    angular_deflection: float = 0.5,
) -> str:

    shape = load_step(step_path)
    mesh = build_mesh_for_shape(shape,mesh_deflection=mesh_deflection,angular_deflection=angular_deflection)
    out_path = Path(stl_path) if stl_path else Path(step_path).with_suffix(".stl")
    mesh.triangulate().clean().save(str(out_path))
    print(f"Saved STL: {out_path}")
    return str(out_path)


