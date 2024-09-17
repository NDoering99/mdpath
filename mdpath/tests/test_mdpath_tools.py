import pytest
import subprocess
import os
import glob

def test_gpcr_2D_vis(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    topology = os.path.join(script_dir, "test_topology.pdb")
    cluster_file = os.path.join(script_dir, "cluster_pathways_dict.pkl")
    cutoff_percentage = "1"

    result = subprocess.run(
        [
            "mdpath_gpcr_image",  
            "-top", topology,
            "-clust", cluster_file,
            "-cut", cutoff_percentage
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0

    generated_files = glob.glob(os.path.join(tmp_path, "GPCR_2D_pathways_cluster*"))
    
    assert len(generated_files) > 0, "No GPCR_2D_pathways_cluster files were generated."

def test_edit_3D_visualization_json(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    recolor = os.path.join(script_dir, "easy_read_colors.json")
    json = os.path.join(script_dir, "quick_precomputed_clusters_paths.json")

    expected_message = (
        "Recoloring requires a valid -json to recolor."
    ) #FAIL

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-recolor", recolor,
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )
    
    assert expected_message in result.stdout

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-recolor", recolor,
            "-json", json
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    assert result.returncode == 0  

    generated_files = glob.glob(os.path.join(script_dir, "*_recolored*"))
    
    assert len(generated_files) > 0, "No recolored json file was generated."

    os.remove(os.path.join(script_dir, "quick_precomputed_clusters_paths_recolored.json"))

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-scale", "3",
            "-json", json,
            "-flat", "3",

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-scale", "3",
            "-json", json,
            "-clusterscale" "3",

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-scale", "3",
            "-json", json,

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

    generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
    
    assert len(generated_files) > 0, "No rescaled json file was generated."

    os.remove(os.path.join(script_dir, "quick_precomputed_clusters_paths_scaled_3.0.json"))

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-flat", "3",
            "-json", json,
            "-scale", "3",

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-flat", "3",
            "-json", json,
            "-clusterscale", "3",

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-flat", "3",
            "-json", json,

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
    
    assert len(generated_files) > 0, "No flattend json file was generated."

    os.remove(os.path.join(script_dir, "quick_precomputed_clusters_paths_flat_3.0.json"))

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-clusterscale", "3",
            "-json", json,
             "-flat", "3",


        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-clusterscale", "3",
            "-json", json,
             "-scale", "3",


        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
    
    assert len(generated_files) == 0, "Files should not be created!"

    result = subprocess.run(
        [
            "mdpath_json_editor",  
            "-clusterscale", "3",
            "-json", json

        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
    
    assert len(generated_files) > 0, "No cluster-scaled json file was generated."

    os.remove(os.path.join(script_dir, "quick_precomputed_clusters_paths_cluster_scaled_3.0.json"))

def test_path_comparison(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    atop = os.path.join(script_dir, "residue_coordinates.pkl")
    bcluster = os.path.join(script_dir, "cluster_pathways_dict_testpc.pkl")
    expected_message = (
        "Topology (residue_coordinates) and bcluster (cluster) are required and a json needed for comparing two simulations."
    ) #FAIL

    result = subprocess.run(
        [
            "mdpath_compare",  
            "-atop", atop
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )
    
    assert expected_message in result.stdout

    result = subprocess.run(
        [
            "mdpath_compare",  
            "-bcluster", bcluster
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )
    
    assert expected_message in result.stdout

    result = subprocess.run(
        [
            "mdpath_compare", 
            "-atop", atop, 
            "-bcluster", bcluster
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0

    generated_files = glob.glob(os.path.join(tmp_path, "morphed_clusters_paths.json"))
    
    assert len(generated_files) > 0, "No rescaled json file was generated."

def test_multitraj_analysis(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    topology = os.path.join(script_dir, "multitraj.pdb")
    multitraj_1 = os.path.join(script_dir, "top_pathways.pkl")
    

    result = subprocess.run(
        [
            "mdpath_multitraj", 
            "-top", topology, 
            "-multitraj", multitraj_1, multitraj_1
        ],
        cwd=tmp_path,  
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    generated_files = glob.glob(os.path.join(tmp_path, "multitraj_clusters_paths.json"))

    assert len(generated_files) > 0, "No rescaled json file was generated."


