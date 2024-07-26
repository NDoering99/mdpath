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
from multiprocessing import Pool
import mdpath.src
import mdpath.src.structure
import mdpath.src.graph
import mdpath.src.cluster
import mdpath.src.mutual_information
import mdpath.src.visualization
import tempfile
from unittest.mock import MagicMock, Mock, patch, call
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from tqdm import tqdm
from Bio import PDB

import mdpath.src.visualization

# Helper functions
def create_mock_pdb(content: str) -> str:
    """Helper function to create a temporary PDB file."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    with open(tmp_file.name, "w") as f:
        f.write(content)
    return tmp_file.name


def test_mdpath_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mdpath" in sys.modules


def test_calculate_distance():
    atom1 = (0.0, 0.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 0.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(
        expected_distance
    )

    atom1 = (1.0, 0.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(
        expected_distance
    )

    atom1 = (0.0, 1.0, 0.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(
        expected_distance
    )

    atom1 = (0.0, 0.0, 1.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = 1.0
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(
        expected_distance
    )

    atom1 = (1.0, 1.0, 1.0)
    atom2 = (0.0, 0.0, 0.0)
    expected_distance = np.sqrt(3)
    assert mdpath.src.structure.calculate_distance(atom1, atom2) == pytest.approx(
        expected_distance
    )


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
    assert weight == pytest.approx(
        expected_weight
    ), f"Expected weight: {expected_weight}, but got: {weight}"

    source = 1
    target = 3
    expected_path = [1, 3]
    expected_weight = 4.0
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(
        expected_weight
    ), f"Expected weight: {expected_weight}, but got: {weight}"

    source = 2
    target = 4
    expected_path = [2, 4]
    expected_weight = 3.0
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(
        expected_weight
    ), f"Expected weight: {expected_weight}, but got: {weight}"

    source = 1
    target = 2
    expected_path = [1, 2]
    expected_weight = 1.0
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
    assert weight == pytest.approx(
        expected_weight
    ), f"Expected weight: {expected_weight}, but got: {weight}"


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

    expected_data = [(66, 68), (66, 69), (67, 69)]
    expected_df = pd.DataFrame(expected_data, columns=["Residue1", "Residue2"])

    result_tuples = list(result_df.itertuples(index=False, name=None))
    expected_tuples = list(expected_df.itertuples(index=False, name=None))

    assert (
        result_tuples == expected_tuples
    ), f"Expected {expected_tuples} but got {result_tuples}"


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

    expected_data = [(66, 67), (66, 68), (66, 69), (67, 68), (67, 69), (68, 69)]
    expected_df = pd.DataFrame(expected_data, columns=["Residue1", "Residue2"])

    result_tuples = list(result_df.itertuples(index=False, name=None))
    expected_tuples = list(expected_df.itertuples(index=False, name=None))

    assert (
        result_tuples == expected_tuples
    ), f"Expected {expected_tuples} but got {result_tuples}"


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
    expected_edges = {(68, 69), (66, 68), (67, 68), (66, 67), (66, 69), (67, 69)}

    assert set(graph.nodes) == expected_nodes
    assert set(graph.edges) == expected_edges
    assert len(graph.nodes) == len(expected_nodes)
    assert len(graph.edges) == len(expected_edges)


def test_calc_dihedral_angle_movement(mocker):
    mock_universe = MagicMock(spec=mda.Universe)
    mock_residue = MagicMock()
    mock_residue.phi_selection.return_value = "phi_selection_mock"
    mock_universe.residues = [mock_residue] * 10

    mock_dihedral = MagicMock(spec=Dihedral)
    mock_dihedral.run.return_value.results.angles = np.array(
        [[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [20.0, 30.0, 40.0]]
    )
    mocker.patch("mdpath.src.structure.Dihedral", return_value=mock_dihedral)

    residue_index = 0
    i, dihedral_angle_movement = mdpath.src.structure.calc_dihedral_angle_movement(
        residue_index, mock_universe
    )

    expected_movement = np.diff(
        np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [20.0, 30.0, 40.0]]), axis=0
    )

    assert i == residue_index
    np.testing.assert_array_equal(dihedral_angle_movement, expected_movement)


def mock_calc_dihedral_angle_movement(residue_id, traj):
    return residue_id, np.array([1.0, 2.0, 3.0])


@patch(
    "mdpath.src.structure.calc_dihedral_angle_movement",
    side_effect=mock_calc_dihedral_angle_movement,
)
def test_calc_dihedral_angle_movement_wrapper(mock_calc_func):
    mock_universe = Mock(spec=mda.Universe)
    residue_id = 42
    args = (residue_id, mock_universe)
    result = mdpath.src.structure.calc_dihedral_angle_movement_wrapper(args)
    assert result[0] == residue_id
    np.testing.assert_array_equal(result[1], np.array([1.0, 2.0, 3.0]))
    mock_calc_func.assert_called_once_with(residue_id, mock_universe)


def mock_calc_dihedral_angle_movement(residue_id, traj):
    if residue_id == 1:
        return residue_id, np.array([[30], [45]], dtype=np.int64)
    elif residue_id == 2:
        return residue_id, np.array([[50], [65]], dtype=np.int64)
    else:
        return residue_id, np.array([], dtype=np.int64)


def mock_calc_dihedral_angle_movement_wrapper(args):
    res_id, traj = args
    return mock_calc_dihedral_angle_movement(res_id, traj)


@patch("mdpath.src.structure.Pool")
@patch(
    "mdpath.src.structure.calc_dihedral_angle_movement_wrapper",
    side_effect=mock_calc_dihedral_angle_movement_wrapper,
)
@patch("mdpath.src.structure.tqdm", return_value=MagicMock())
def test_calculate_dihedral_movement_parallel(mock_tqdm, mock_wrapper, mock_pool):
    mock_traj = MagicMock()
    mock_pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.imap_unordered.return_value = iter(
        [
            (1, np.array([[30], [45]], dtype=np.int64)),
            (2, np.array([[50], [65]], dtype=np.int64)),
        ]
    )

    df = mdpath.src.structure.calculate_dihedral_movement_parallel(
        num_parallel_processes=2,
        first_res_num=1,
        last_res_num=2,
        num_residues=2,
        traj=mock_traj,
    )

    expected_df = pd.DataFrame({"Res 1": [30, 45], "Res 2": [50, 65]}, dtype=np.int64)

    pd.testing.assert_frame_equal(df, expected_df)

    df_empty = mdpath.src.structure.calculate_dihedral_movement_parallel(
        num_parallel_processes=2,
        first_res_num=1,
        last_res_num=1,
        num_residues=0,
        traj=mock_traj,
    )

    expected_df_empty = pd.DataFrame(dtype=np.int64)
    pd.testing.assert_frame_equal(df_empty, expected_df_empty)

    def error_wrapper(args):
        raise ValueError("Mock error")

    mock_pool_instance.imap_unordered.side_effect = error_wrapper

    df_error = mdpath.src.structure.calculate_dihedral_movement_parallel(
        num_parallel_processes=2,
        first_res_num=1,
        last_res_num=2,
        num_residues=2,
        traj=mock_traj,
    )

    expected_df_error = pd.DataFrame(dtype=np.int64)
    pd.testing.assert_frame_equal(df_error, expected_df_error)


def test_graph_assign_weights():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    data = {
        "Residue Pair": [("Res 1", "Res 2"), ("Res 2", "Res 3")],
        "MI Difference": [0.5, 0.8],
    }
    mi_diff_df = pd.DataFrame(data)

    G_weighted = mdpath.src.graph.graph_assign_weights(G, mi_diff_df)

    expected_weights = {(1, 2): 0.5, (2, 3): 0.8}
    for edge, weight in expected_weights.items():
        assert "weight" in G_weighted.edges[edge], f"Weight for edge {edge} not found"
        assert (
            G_weighted.edges[edge]["weight"] == weight
        ), f"Weight for edge {edge} is not {weight}"
    for edge in G_weighted.edges:
        if edge not in expected_weights:
            assert (
                "weight" not in G_weighted.edges[edge]
            ), f"Unexpected weight for edge {edge}"


def test_collect_path_total_weights():
    G = nx.Graph()
    G.add_edge(1, 2, weight=10)
    G.add_edge(2, 3, weight=20)
    G.add_edge(1, 3, weight=15)

    df = pd.DataFrame({"Residue1": [1, 2], "Residue2": [3, 4]})

    # Test cases
    test_cases = [
        {
            "mock_side_effect": [([1, 2, 3], 30), nx.NetworkXNoPath],
            "expected_result": [([1, 2, 3], 30)],
        },
        {
            "mock_side_effect": [nx.NetworkXNoPath, nx.NetworkXNoPath],
            "expected_result": [],
        },
        {
            "mock_side_effect": [([1, 2, 3], 30), nx.NetworkXNoPath],
            "df": pd.DataFrame(columns=["Residue1", "Residue2"]),
            "expected_result": [],
        },
    ]

    for case in test_cases:
        with patch(
            "mdpath.src.graph.max_weight_shortest_path"
        ) as mock_max_weight_shortest_path:
            mock_max_weight_shortest_path.side_effect = case["mock_side_effect"]

            df_test = case.get("df", df)
            result = mdpath.src.graph.collect_path_total_weights(G, df_test)

            assert result == case["expected_result"]


def test_nmi_calc(mocker):
    data = {
        "Residue1": np.random.uniform(-180, 180, size=100),
        "Residue2": np.random.uniform(-180, 180, size=100),
        "Residue3": np.random.uniform(-180, 180, size=100),
    }
    df_all_residues = pd.DataFrame(data)

    result = mdpath.src.mutual_information.NMI_calc(df_all_residues)

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    assert (
        "Residue Pair" in result.columns
    ), "DataFrame should contain 'Residue Pair' column"
    assert (
        "MI Difference" in result.columns
    ), "DataFrame should contain 'MI Difference' column"

    assert not result.empty, "DataFrame should not be empty"

    expected_shape = (
        len(df_all_residues.columns) * (len(df_all_residues.columns) - 1),
        2,
    )
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"

    assert (
        result["MI Difference"] >= 0
    ).all(), "MI Difference values should be non-negative"


def test_calculate_overlap_for_pathway():
    df_basic = pd.DataFrame({"Residue1": [1, 2, 3, 4], "Residue2": [2, 3, 4, 5]})

    args_basic = (0, [1, 2], [[1, 2], [3, 4], [1, 4]], df_basic)
    expected_basic = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 1},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 1},
        {"Pathway1": 0, "Pathway2": 2, "Overlap": 1},
        {"Pathway1": 2, "Pathway2": 0, "Overlap": 1},
    ]
    result_basic = mdpath.src.cluster.calculate_overlap_for_pathway(args_basic)
    assert result_basic == expected_basic

    args_no_overlap = (0, [1, 2], [[5, 6], [7, 8]], df_basic)
    expected_no_overlap = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 0},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 0},
    ]
    result_no_overlap = mdpath.src.cluster.calculate_overlap_for_pathway(
        args_no_overlap
    )
    assert result_no_overlap == expected_no_overlap

    args_edge_case = (0, [10, 11], [[20, 21], [30, 31]], df_basic)
    expected_edge_case = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 0},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 0},
    ]
    result_edge_case = mdpath.src.cluster.calculate_overlap_for_pathway(args_edge_case)
    assert result_edge_case == expected_edge_case

    df_empty = pd.DataFrame(columns=["Residue1", "Residue2"])
    args_empty_df = (0, [1, 2], [[3, 4], [5, 6]], df_empty)
    expected_empty_df = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 0},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 0},
    ]
    result_empty_df = mdpath.src.cluster.calculate_overlap_for_pathway(args_empty_df)
    assert result_empty_df == expected_empty_df


@patch("mdpath.src.cluster.calculate_overlap_parallel")
def test_calculate_overlap_parallel(mock_calculate_overlap_parallel):
    pathways = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame({"Residue1": [1, 2, 3], "Residue2": [2, 3, 4]})
    num_processes = 2

    # Expected result
    expected_data = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 1},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 1},
        {"Pathway1": 0, "Pathway2": 2, "Overlap": 0},
        {"Pathway1": 2, "Pathway2": 0, "Overlap": 0},
        {"Pathway1": 1, "Pathway2": 2, "Overlap": 1},
        {"Pathway1": 2, "Pathway2": 1, "Overlap": 1},
    ]
    expected_df = pd.DataFrame(expected_data)
    mock_calculate_overlap_parallel.return_value = expected_df
    result_df = mdpath.src.cluster.calculate_overlap_parallel(
        pathways, df, num_processes
    )
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_pathways_cluster(tmp_path):
    data = {
        "Pathway1": [
            "A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D",
            "E", "E", "E", "E", "F", "F", "F", "F", "G", "G", "G", "G", "H", "H", "H", "H"
        ],
        "Pathway2": [
            "A", "B", "C", "D", "B", "C", "D", "E", "C", "D", "E", "F", "D", "E", "F", "G",
            "E", "F", "G", "H", "F", "G", "H", "A", "G", "H", "A", "B", "H", "A", "B", "C"
        ],
        "Overlap": [
            1, 8, 5, 2,  8, 1, 6, 2,  5, 6, 1, 7,  2, 2, 7, 3,  
            2, 7, 3, 4,  7, 3, 4, 5,  3, 4, 5, 6,  4, 5, 6, 1
        ],  
    }
    overlap_df = pd.DataFrame(data)

    save_path = tmp_path / "clustered_paths.png"
    clusters = mdpath.src.cluster.pathways_cluster(overlap_df, save_path=str(save_path))
    assert isinstance(clusters, dict), "Output is not a dictionary"
    assert save_path.exists(), "Dendrogram image was not saved"
    assert all(len(pathways) > 0 for pathways in clusters.values()), "One of the clusters is empty"
    all_pathways = [item for sublist in clusters.values() for item in sublist]
    expected_pathways = sorted(set(data["Pathway1"] + data["Pathway2"]))
    assert sorted(all_pathways) == expected_pathways, "Clusters do not contain the expected pathways"
    num_clusters = len(clusters)
    assert num_clusters >= 2, "Number of clusters should be at least 2"

def are_vectors_close(vec1, vec2, tol=1e-6):
    return np.allclose(vec1, vec2, atol=tol)

def test_residue_CA_coordinates():
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
    end_residue = 69

    expected_result = {
        66: [[162.030, 133.517, 139.402]],
        67: [[162.863, 134.856, 135.941]],
        68: [[163.415, 131.357, 134.496]],
        69: [[160.061, 130.137, 135.858]]
    }

    result = mdpath.src.visualization.residue_CA_coordinates(pdb_file, end_residue)
    result = {
        k: [list(v) for v in vs]  
        for k, vs in result.items()
    }

    for res_id, coords_list in expected_result.items():
        assert res_id in result, f"Residue {res_id} not found in result"
        for expected_coords, actual_coords in zip(coords_list, result[res_id]):
            assert are_vectors_close(expected_coords, actual_coords), \
                f"Expected {expected_coords} but got {actual_coords} for residue {res_id}"


def test_apply_backtracking():
    original_dict = {
        'cluster1': [[1, 2], [3, 4]],
        'cluster2': [[5, 6], [7, 8]],
        'cluster3': [[9, 10]]
    }
    translation_dict = {
        1: 'A',
        2: 'B',
        5: 'E',
        6: 'F',
        10: 'J'
    }
    expected = {
        'cluster1': [['A', 'B'], [3, 4]],
        'cluster2': [['E', 'F'], [7, 8]],
        'cluster3': [[9, 'J']]
    }

    result = mdpath.src.visualization.apply_backtracking(original_dict, translation_dict)
    assert result == expected

