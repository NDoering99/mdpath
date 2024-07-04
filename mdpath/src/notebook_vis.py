import json
import nglview as nv


def load_precomputed_data(json_path: str) -> dict:
    """Loads precomputed cluster properties from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing precomputed cluster properties.

    Returns:
        dict: Dictionary containing precomputed cluster properties for visualization
    """
    with open(json_path, "r") as json_file:
        precomputed_data = json.load(json_file)
    return precomputed_data


def generate_ngl_script(precomputed_data: dict, view: nv.NGLWidget) -> None:
    """Generates NGL script and edits view for visualizing precomputed cluster pathways as cones between residues.

    Args:
        precomputed_data (dict): Precomputed data for visualization.
        view (nv.NGLWidget): NGL view object.

    Returns:
        None: Only edits the view object.
    """
    pathways = {}

    for prop in precomputed_data:
        clusterid = prop["clusterid"]
        pathway_index = prop["pathway_index"]
        key = (clusterid, pathway_index)

        if key not in pathways:
            pathways[key] = []

        shape_segment = f"""
            shape.addCylinder([{prop["coord1"][0]}, {prop["coord1"][1]}, {prop["coord1"][2]}], 
                              [{prop["coord2"][0]}, {prop["coord2"][1]}, {prop["coord2"][2]}], 
                              [{prop["color"][0]}, {prop["color"][1]}, {prop["color"][2]}], 
                              {prop["radius"]});
        """
        pathways[key].append(shape_segment)

    for (clusterid, pathway_index), shape_segments in pathways.items():
        shape_script = f"""
            var shape = new NGL.Shape('Cluster{clusterid}_Pathway{pathway_index}');
            {"".join(shape_segments)}
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation('buffer');
        """
        if view:
            view._execute_js_code(shape_script)
        else:
            print("View is not defined.")


def generate_cluster_ngl_script(precomputed_data: dict, view: nv.NGLWidget) -> None:
    """Generates NGL script and edits view for visualizing precomputed cluster pathways as cones between residues.

    Args:
        precomputed_data (dict): Precomputed data for visualization.
        view (nv.NGLWidget): NGL view object.

    Returns:
        None: Only edits the view object.
    """
    cluster_shapes = {}

    for prop in precomputed_data:
        clusterid = prop["clusterid"]
        if clusterid not in cluster_shapes:
            cluster_shapes[clusterid] = []

        shape_segment = f"""
            shape.addCylinder([{prop["coord1"][0]}, {prop["coord1"][1]}, {prop["coord1"][2]}], 
                              [{prop["coord2"][0]}, {prop["coord2"][1]}, {prop["coord2"][2]}], 
                              [{prop["color"][0]}, {prop["color"][1]}, {prop["color"][2]}], 
                              {prop["radius"]});
        """
        cluster_shapes[clusterid].append(shape_segment)

    for clusterid, shape_segments in cluster_shapes.items():
        shape_script = f"""
            var shape = new NGL.Shape('Cluster{clusterid}');
            {"".join(shape_segments)}
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation('buffer');
        """
        if view:
            view._execute_js_code(shape_script)
        else:
            print("View is not defined.")
