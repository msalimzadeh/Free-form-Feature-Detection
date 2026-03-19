import sys
import argparse

import pyvista as pv
import numpy as np

from OCP.STEPControl import STEPControl_Reader
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

    if not picked_points or not picked_cells:
        return None, None, None

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
    
    reader = STEPControl_Reader()
    reader.ReadFile(step_path)
    reader.TransferRoots()
    shape = reader.OneShape()

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
    mesh = BRepMesh_IncrementalMesh(
        shape, mesh_deflection, False, angular_deflection, True
    )
    mesh.Perform()
    if not mesh.IsDone():
        print("BRepMesh_IncrementalMesh failed", file=sys.stderr)
        return None

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

    if not all_verts:
        print("No triangulation produced.", file=sys.stderr)
        return None

    verts = np.array(all_verts, dtype=float)
    cells = np.array(all_cells, dtype=np.int64)
    face_ids = np.array(all_face_ids, dtype=np.int32)

    poly = pv.PolyData(verts, cells)
    poly.cell_data["face_id"] = face_ids
    return poly

# Volume removal visualization
def build_mesh_for_shape(shape, mesh_deflection= 0.001, angular_deflection = 0.5):

    faces_list: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        faces_list.append(face)
        exp.Next()
    if not faces_list:
        return None
    return _build_face_mesh(
        shape, faces_list,
        mesh_deflection=mesh_deflection,
        angular_deflection=angular_deflection,
    )


def pick_brep_face( shape, faces_list, mesh_deflection = 0.001):
    mesh = _build_face_mesh(shape, faces_list, mesh_deflection=mesh_deflection)
    if mesh is None:
        return None
    face_id, cell_id, point = pick_face_on_mesh(mesh)
    if face_id is None:
        return None
    if 0 <= face_id < len(faces_list):
        return faces_list[face_id]
    return None


def visualize_faces_on_mesh(shape, faces_list, selected_face_ids, mesh_deflection = 0.001):
    mesh = _build_face_mesh(shape, faces_list, mesh_deflection=mesh_deflection)
    if mesh is None:
        return
    if not selected_face_ids:
        selected_ids_set = set()
    else:
        selected_ids_set = set(int(i) for i in selected_face_ids)

    face_ids = mesh.cell_data.get("face_id")
    if face_ids is None:
        return

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


def main():
    parser = argparse.ArgumentParser(description="Read STEP and expose faces")
    parser.add_argument("--step", required=True, help="Path to STEP file")
    args = parser.parse_args()

    step_path = args.step
    shape, faces = read_step_from_user(step_path)
    face = pick_brep_face(shape, faces)
    if face is None:
        print("No face picked.")
    else:
        print(face)


if __name__ == "__main__":
    raise SystemExit(main())
