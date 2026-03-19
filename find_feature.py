from config import *
from graph import *
from region_growing import grow_region
from extrude_feature import *

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect free-form feature, extrude it, and intersect with stock removal volume."
    )
    parser.add_argument("--step", required=True, help="Path to finished part STEP file")
    parser.add_argument("--stock", help="Path to stock STEP file")
    parser.add_argument(
        "--out",
        default="feature_removal.step",
        help="Output STEP file for intersected feature removal volume",
    )
    args = parser.parse_args()

    step_path = args.step
    if args.stock:
        stock_path = args.stock
    else:
        stock_path = None

    # Load part and stock shapes
    shape, faces_list = read_step_from_user(step_path)  # finished part
    if stock_path:
        stock_shape = read_step_solid(stock_path)
    else:
        stock_shape = None
    face_id_map = iter_faces(faces_list)
    G = build_face_adjacency(faces_list)    
    attach_face_attributes(G, faces_list)
    attach_edge_angles(G, faces_list)
    # print_face_table(G)
    # plot_graph(G, title=f"Face adjacency: {step_path}")
    # visualize_3d(shape, faces_list, face_id_map, mesh_deflection=0.001)
    # graph_print(G)
    picked_face = pick_brep_face(shape, faces_list)
    face_id = faces_list.index(picked_face)
    G.nodes[face_id]["flag"] = True
    # print(f"\nPicked face_id={face_id}: flag set to {G.nodes[face_id]['flag']}")

    feature_faces = grow_region(G, face_id)
    # graph_print(G)s
    # print("Feature region face IDs:", feature_faces)

    # Visualize the CAD model with the feature_faces region highlighted
    visualize_faces_on_mesh(shape, faces_list, selected_face_ids=list(feature_faces))
    detected_patch_faces = [faces_list[i] for i in feature_faces]

    # Extrude the detected patch to obtain a feature volume (prism)
    removal_prism = extrude_feature_patch(
        detected_patch_faces,
        direction=(0,0,1),   # machining direction
        length=20,          # large enough to reach stock top
    )

    # Compute the true feature removal volume:
    # (stock - part) ∩ removal_prism
    # feature_removal = compute_feature_removal_volume(
    #     stock_shape=stock_shape,
    #     part_shape=shape,
    #     feature_shape=removal_prism,
    #     mesh_deflection=0.2,
    # )

    # Visualize the intersected removal volume
    feature_mesh = build_mesh_for_shape(removal_prism, mesh_deflection=0.001)
    if feature_mesh is not None:
        plotter = pv.Plotter()
        plotter.add_mesh(feature_mesh, color="gray")
        plotter.show()

    # Export intersected feature removal to STEP
    output_step = args.out
    save_shape_to_step(removal_prism, output_step)
    print(f"Feature removal volume saved to STEP file: {output_step}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
