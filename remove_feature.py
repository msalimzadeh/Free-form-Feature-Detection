from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
import trimesh

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


def removal_mesh(extrusion, ips, out):
    extrusion = trimesh.load(extrusion)
    ips = trimesh.load(ips)
    intersection = extrusion.intersection(ips)
    intersection.export(out)


