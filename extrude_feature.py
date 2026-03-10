from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCP.gp import gp_Vec
from OCP.TopoDS import TopoDS_Shell, TopoDS_Shape
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCP.BRepMesh import BRepMesh_IncrementalMesh

def extrude_feature_patch(feature_faces, direction=(0,0,1), length=200):

    sewing = BRepBuilderAPI_Sewing()
    for f in feature_faces:
        sewing.Add(f)

    sewing.Perform()
    shell = sewing.SewedShape()

    vec = gp_Vec(direction[0], direction[1], direction[2])
    vec.Multiply(length)

    prism = BRepPrimAPI_MakePrism(shell, vec).Shape()

    return prism




def read_step_solid(path: str) -> TopoDS_Shape:
    """Utility: read a STEP file and return a solid shape."""
    r = STEPControl_Reader()
    status = r.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    r.TransferRoots()
    return r.OneShape()   # assume it's a solid (or compound with one solid)


def compute_feature_removal_volume(
    stock_shape: TopoDS_Shape,
    part_shape: TopoDS_Shape,
    feature_shape: TopoDS_Shape,
    mesh_deflection: float = 0.2,
) -> TopoDS_Shape:

    # 1) Compute removal volume: stock minus part.
    cut_op = BRepAlgoAPI_Cut(stock_shape, part_shape)
    cut_op.Build()
    if not cut_op.IsDone():
        raise RuntimeError(
            "Boolean cut (stock minus part) failed; check that both shapes are valid solids."
        )
    removal_volume = cut_op.Shape()

    # 2) Intersect removal volume with the feature volume.
    common_op = BRepAlgoAPI_Common(removal_volume, feature_shape)
    common_op.Build()
    if not common_op.IsDone():
        raise RuntimeError(
            "Boolean common between removal volume and feature failed; check input shapes."
        )
    final_removal = common_op.Shape()

    # 3) Mesh the final removal volume (attaches triangulations to faces).
    BRepMesh_IncrementalMesh(final_removal, mesh_deflection).Perform()

    return final_removal


def save_shape_to_step(shape: TopoDS_Shape, path: str) -> None:
    """Write a TopoDS_Shape to a STEP file."""
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP file to {path}")




