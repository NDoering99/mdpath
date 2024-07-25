"""
Unit and regression test for the mdpath package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np
import pandas as pd
import pytest
import networkx as nx
import mdpath
import mdpath.src
import mdpath.src.structure
import mdpath.src.graph
import tempfile 
from unittest.mock import MagicMock
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral

#Helper functions
def create_mock_pdb(content: str) -> str:
    """Helper function to create a temporary PDB file."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    with open(tmp_file.name, 'w') as f:
        f.write(content)
    return tmp_file.name

def test_mdpath_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mdpath" in sys.modules


def test_calculate_distance():
    atom1 = (0.0, 0.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 0.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(expected_distance)

    atom1 = (1.0, 0.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(expected_distance)

    atom1 = (0.0, 1.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(expected_distance)

    atom1 = (0.0, 0.0, 1.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(expected_distance)

    atom1 = (1.0, 1.0, 1.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = np.sqrt(3)
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(expected_distance)


def test_max_weight_shortest_path():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(1, 3, weight=4.0)
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(2, 4, weight=3.0)
    
    source = 1
    target = 4
    expected_path = [1, 3, 4]
    expected_weight = 5.0
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(expected_weight), f"Expected weight: {expected_weight}, but got: {weight}"

    source = 1
    target = 3
    expected_path = [1, 3]
    expected_weight = 4.0
    path, weight =  mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(expected_weight), f"Expected weight: {expected_weight}, but got: {weight}"
    
    source = 2
    target = 4
    expected_path = [2, 4]
    expected_weight = 3.0
    path, weight =  mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(expected_weight), f"Expected weight: {expected_weight}, but got: {weight}"

    source = 1
    target = 2
    expected_path = [1, 2]
    expected_weight = 1.0
    path, weight =  mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(expected_weight), f"Expected weight: {expected_weight}, but got: {weight}"


def test_collect_path_total_weights():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(1, 3, weight=4.0)
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(2, 4, weight=3.0)
    
    data = {
        'Residue1': [1, 1, 2],
        'Residue2': [3, 4, 4]
    }
    df_distant_residues = pd.DataFrame(data)
    
    expected_results = [
        ([1, 3], 4.0), 
        ([1, 2, 4], 4.0),  
        ([2, 4], 3.0)  
    ]
    
    results = mdpath.src.graph.collect_path_total_weights(G, df_distant_residues)
    
    assert len(results) == len(expected_results)
    for result, expected in zip(results, expected_results):
        path, weight = result
        expected_path, expected_weight = expected
        assert path == expected_path
        assert weight == pytest.approx(expected_weight)


def test_res_num_from_pdb():
    pdb_content = """
ATOM      1  N   SER R  66     163.079 132.512 139.525  1.00 67.17           N
ATOM      2  CA  SER R  66     162.030 133.517 139.402  1.00 67.17           C
ATOM      3  C   SER R  66     161.837 133.927 137.947  1.00 67.17           C
ATOM      4  O   SER R  66     160.746 133.785 137.394  1.00 67.17           O
ATOM      5  CB  SER R  66     162.361 134.744 140.256  1.00 67.17           C
ATOM      6  OG  SER R  66     163.513 135.410 139.767  1.00 67.17           O
ATOM      7  N   MET R  67     162.906 134.456 137.345  1.00 67.62           N
ATOM      8  CA  MET R  67     162.863 134.856 135.941  1.00 67.62           C
ATOM      9  C   MET R  67     162.712 133.650 135.020  1.00 67.62           C
ATOM     10  O   MET R  67     161.925 133.688 134.064  1.00 67.62           O
ATOM     11  CB  MET R  67     164.124 135.646 135.590  1.00 67.62           C
ATOM     12  CG  MET R  67     164.130 136.230 134.189  1.00 67.62           C
ATOM     13  SD  MET R  67     162.847 137.472 133.944  1.00 67.62           S
ATOM     14  CE  MET R  67     163.512 138.833 134.901  1.00 67.62           C
ATOM     15  N   ILE R  68     163.445 132.569 135.314  1.00 62.30           N
ATOM     16  CA  ILE R  68     163.415 131.357 134.496  1.00 62.30           C
ATOM     17  C   ILE R  68     162.033 130.714 134.535  1.00 62.30           C
ATOM     18  O   ILE R  68     161.528 130.240 133.509  1.00 62.30           O
ATOM     19  CB  ILE R  68     164.513 130.386 134.971  1.00 62.30           C
ATOM     20  CG1 ILE R  68     165.891 131.038 134.844  1.00 62.30           C
ATOM     21  CG2 ILE R  68     164.493 129.087 134.176  1.00 62.30           C
ATOM     22  CD1 ILE R  68     166.997 130.265 135.529  1.00 62.30           C
ATOM     23  N   THR R  69     161.394 130.716 135.710  1.00 57.94           N
ATOM     24  CA  THR R  69     160.061 130.137 135.858  1.00 57.94           C
ATOM     25  C   THR R  69     159.027 130.903 135.036  1.00 57.94           C
ATOM     26  O   THR R  69     158.209 130.295 134.331  1.00 57.94           O
ATOM     27  CB  THR R  69     159.673 130.122 137.339  1.00 57.94           C
TER
"""
    pdb_file = create_mock_pdb(pdb_content)
    assert mdpath.src.structure.res_num_from_pdb(pdb_file) == (66, 69)

def test_faraway_residues():
    pdb_content = """
ATOM      1  N   SER R  66     163.079 132.512 139.525  1.00 67.17           N
ATOM      2  CA  SER R  66     162.030 133.517 139.402  1.00 67.17           C
ATOM      3  C   SER R  66     161.837 133.927 137.947  1.00 67.17           C
ATOM      4  O   SER R  66     160.746 133.785 137.394  1.00 67.17           O
ATOM      5  CB  SER R  66     162.361 134.744 140.256  1.00 67.17           C
ATOM      6  OG  SER R  66     163.513 135.410 139.767  1.00 67.17           O
ATOM      7  N   MET R  67     162.906 134.456 137.345  1.00 67.62           N
ATOM      8  CA  MET R  67     162.863 134.856 135.941  1.00 67.62           C
ATOM      9  C   MET R  67     162.712 133.650 135.020  1.00 67.62           C
ATOM     10  O   MET R  67     161.925 133.688 134.064  1.00 67.62           O
ATOM     11  CB  MET R  67     164.124 135.646 135.590  1.00 67.62           C
ATOM     12  CG  MET R  67     164.130 136.230 134.189  1.00 67.62           C
ATOM     13  SD  MET R  67     162.847 137.472 133.944  1.00 67.62           S
ATOM     14  CE  MET R  67     163.512 138.833 134.901  1.00 67.62           C
ATOM     15  N   ILE R  68     163.445 132.569 135.314  1.00 62.30           N
ATOM     16  CA  ILE R  68     163.415 131.357 134.496  1.00 62.30           C
ATOM     17  C   ILE R  68     162.033 130.714 134.535  1.00 62.30           C
ATOM     18  O   ILE R  68     161.528 130.240 133.509  1.00 62.30           O
ATOM     19  CB  ILE R  68     164.513 130.386 134.971  1.00 62.30           C
ATOM     20  CG1 ILE R  68     165.891 131.038 134.844  1.00 62.30           C
ATOM     21  CG2 ILE R  68     164.493 129.087 134.176  1.00 62.30           C
ATOM     22  CD1 ILE R  68     166.997 130.265 135.529  1.00 62.30           C
ATOM     23  N   THR R  69     161.394 130.716 135.710  1.00 57.94           N
ATOM     24  CA  THR R  69     160.061 130.137 135.858  1.00 57.94           C
ATOM     25  C   THR R  69     159.027 130.903 135.036  1.00 57.94           C
ATOM     26  O   THR R  69     158.209 130.295 134.331  1.00 57.94           O
ATOM     27  CB  THR R  69     159.673 130.122 137.339  1.00 57.94           C
TER
"""
    pdb_file = create_mock_pdb(pdb_content)
    result_df = mdpath.src.structure.faraway_residues(pdb_file, end=69, dist=1.5)
    
    expected_data = [
        (66, 68),
        (66, 69),
        (67, 69)
    ]
    expected_df = pd.DataFrame(expected_data, columns=["Residue1", "Residue2"])
    
    result_tuples = list(result_df.itertuples(index=False, name=None))
    expected_tuples = list(expected_df.itertuples(index=False, name=None))
    
    assert result_tuples == expected_tuples, f"Expected {expected_tuples} but got {result_tuples}"

def test_close_residues():
    pdb_content = """
ATOM      1  N   SER R  66     163.079 132.512 139.525  1.00 67.17           N
ATOM      2  CA  SER R  66     162.030 133.517 139.402  1.00 67.17           C
ATOM      3  C   SER R  66     161.837 133.927 137.947  1.00 67.17           C
ATOM      4  O   SER R  66     160.746 133.785 137.394  1.00 67.17           O
ATOM      5  CB  SER R  66     162.361 134.744 140.256  1.00 67.17           C
ATOM      6  OG  SER R  66     163.513 135.410 139.767  1.00 67.17           O
ATOM      7  N   MET R  67     162.906 134.456 137.345  1.00 67.62           N
ATOM      8  CA  MET R  67     162.863 134.856 135.941  1.00 67.62           C
ATOM      9  C   MET R  67     162.712 133.650 135.020  1.00 67.62           C
ATOM     10  O   MET R  67     161.925 133.688 134.064  1.00 67.62           O
ATOM     11  CB  MET R  67     164.124 135.646 135.590  1.00 67.62           C
ATOM     12  CG  MET R  67     164.130 136.230 134.189  1.00 67.62           C
ATOM     13  SD  MET R  67     162.847 137.472 133.944  1.00 67.62           S
ATOM     14  CE  MET R  67     163.512 138.833 134.901  1.00 67.62           C
ATOM     15  N   ILE R  68     163.445 132.569 135.314  1.00 62.30           N
ATOM     16  CA  ILE R  68     163.415 131.357 134.496  1.00 62.30           C
ATOM     17  C   ILE R  68     162.033 130.714 134.535  1.00 62.30           C
ATOM     18  O   ILE R  68     161.528 130.240 133.509  1.00 62.30           O
ATOM     19  CB  ILE R  68     164.513 130.386 134.971  1.00 62.30           C
ATOM     20  CG1 ILE R  68     165.891 131.038 134.844  1.00 62.30           C
ATOM     21  CG2 ILE R  68     164.493 129.087 134.176  1.00 62.30           C
ATOM     22  CD1 ILE R  68     166.997 130.265 135.529  1.00 62.30           C
ATOM     23  N   THR R  69     161.394 130.716 135.710  1.00 57.94           N
ATOM     24  CA  THR R  69     160.061 130.137 135.858  1.00 57.94           C
ATOM     25  C   THR R  69     159.027 130.903 135.036  1.00 57.94           C
ATOM     26  O   THR R  69     158.209 130.295 134.331  1.00 57.94           O
ATOM     27  CB  THR R  69     159.673 130.122 137.339  1.00 57.94           C
TER
"""
    pdb_file = create_mock_pdb(pdb_content)
    result_df = mdpath.src.structure.close_residues(pdb_file, end=69, dist=10.0)
    
    expected_data = [
        (66, 67),
        (66, 68),
        (66, 69),
        (67, 68),
        (67, 69),
        (68, 69)
    ]
    expected_df = pd.DataFrame(expected_data, columns=["Residue1", "Residue2"])
    
    result_tuples = list(result_df.itertuples(index=False, name=None))
    expected_tuples = list(expected_df.itertuples(index=False, name=None))
    
    assert result_tuples == expected_tuples, f"Expected {expected_tuples} but got {result_tuples}"


def test_graph_building():
    pdb_content = """
ATOM      1  N   SER R  66     163.079 132.512 139.525  1.00 67.17           N
ATOM      2  CA  SER R  66     162.030 133.517 139.402  1.00 67.17           C
ATOM      3  C   SER R  66     161.837 133.927 137.947  1.00 67.17           C
ATOM      4  O   SER R  66     160.746 133.785 137.394  1.00 67.17           O
ATOM      5  CB  SER R  66     162.361 134.744 140.256  1.00 67.17           C
ATOM      6  OG  SER R  66     163.513 135.410 139.767  1.00 67.17           O
ATOM      7  N   MET R  67     162.906 134.456 137.345  1.00 67.62           N
ATOM      8  CA  MET R  67     162.863 134.856 135.941  1.00 67.62           C
ATOM      9  C   MET R  67     162.712 133.650 135.020  1.00 67.62           C
ATOM     10  O   MET R  67     161.925 133.688 134.064  1.00 67.62           O
ATOM     11  CB  MET R  67     164.124 135.646 135.590  1.00 67.62           C
ATOM     12  CG  MET R  67     164.130 136.230 134.189  1.00 67.62           C
ATOM     13  SD  MET R  67     162.847 137.472 133.944  1.00 67.62           S
ATOM     14  CE  MET R  67     163.512 138.833 134.901  1.00 67.62           C
ATOM     15  N   ILE R  68     163.445 132.569 135.314  1.00 62.30           N
ATOM     16  CA  ILE R  68     163.415 131.357 134.496  1.00 62.30           C
ATOM     17  C   ILE R  68     162.033 130.714 134.535  1.00 62.30           C
ATOM     18  O   ILE R  68     161.528 130.240 133.509  1.00 62.30           O
ATOM     19  CB  ILE R  68     164.513 130.386 134.971  1.00 62.30           C
ATOM     20  CG1 ILE R  68     165.891 131.038 134.844  1.00 62.30           C
ATOM     21  CG2 ILE R  68     164.493 129.087 134.176  1.00 62.30           C
ATOM     22  CD1 ILE R  68     166.997 130.265 135.529  1.00 62.30           C
ATOM     23  N   THR R  69     161.394 130.716 135.710  1.00 57.94           N
ATOM     24  CA  THR R  69     160.061 130.137 135.858  1.00 57.94           C
ATOM     25  C   THR R  69     159.027 130.903 135.036  1.00 57.94           C
ATOM     26  O   THR R  69     158.209 130.295 134.331  1.00 57.94           O
ATOM     27  CB  THR R  69     159.673 130.122 137.339  1.00 57.94           C
TER
"""
    pdb_file = create_mock_pdb(pdb_content)
    graph = mdpath.src.graph.graph_building(pdb_file, end=69)
    expected_nodes = {66, 67, 68, 69}
    expected_edges = {
        (68, 69), (66, 68), (67, 68), (66, 67), (66, 69), (67, 69)   
    }

    assert set(graph.nodes) == expected_nodes
    assert set(graph.edges) == expected_edges
    assert len(graph.nodes) == len(expected_nodes)
    assert len(graph.edges) == len(expected_edges)


def test_update_progress():
    mock_tqdm = MagicMock()
    result = mdpath.src.structure.update_progress(mock_tqdm)
    mock_tqdm.update.assert_called_once()
    assert result == mock_tqdm

def test_calc_dihedral_angle_movement(mocker):
    mock_universe = MagicMock(spec=mda.Universe)
    mock_residue = MagicMock()
    mock_residue.phi_selection.return_value = 'phi_selection_mock'
    mock_universe.residues = [mock_residue] * 10  

    mock_dihedral = MagicMock(spec=Dihedral)
    mock_dihedral.run.return_value.results.angles = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ])
    mocker.patch('mdpath.src.structure.Dihedral', return_value=mock_dihedral)

    residue_index = 0
    i, dihedral_angle_movement = mdpath.src.structure.calc_dihedral_angle_movement(residue_index, mock_universe)
    
    expected_movement = np.diff(np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ]), axis=0)
    
    assert i == residue_index
    np.testing.assert_array_equal(dihedral_angle_movement, expected_movement)

