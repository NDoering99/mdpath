import json
import nglview as nv


class NotebookVisualization:
    def __init__(self, pdb_path, json_path) -> None:
        self.pdb_path = pdb_path
        self.view = self.load_ngl_view()
        self.json_path = json_path
        self.precomputed_data = self.load_precomputed_data()

    def load_ngl_view(self):
        view = nv.show_file(self.pdb_path)
        view.display(gui=True, style="ngl")
        return view

    def load_precomputed_data(self) -> dict:
        """Loads precomputed cluster properties from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing precomputed cluster properties.

        Returns:
            dict: Dictionary containing precomputed cluster properties for visualization
        """
        with open(self.json_path, "r") as json_file:
            precomputed_data = json.load(json_file)
        return precomputed_data


    def generate_cluster_ngl_script(self) -> None:
        """Generates NGL script and edits view for visualizing precomputed cluster pathways as cones between residues.

        Args:
            precomputed_data (dict): Precomputed data for visualization.
            view (nv.NGLWidget): NGL view object.

        Returns:
            None: Only edits the view object.
        """
        cluster_shapes = {}

        for prop in self.precomputed_data:
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
            if self.view:
                self.view._execute_js_code(shape_script)
            else:
                print("View is not defined.")
