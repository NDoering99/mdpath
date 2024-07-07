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
    expected_path = [1, 2, 4]
    expected_weight = 4.0  
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path
    assert weight == pytest.approx(expected_weight)

    source = 1
    target = 3
    expected_path = [1, 3]
    expected_weight = 4.0 
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path
    assert weight == pytest.approx(expected_weight)
 
    source = 1
    target = 4
    expected_path = [1, 3, 4]
    expected_weight = 5.0  
    path, weight = mdpath.src.graph.max_weight_shortest_path(G, source, target)
    assert path == expected_path
    assert weight == pytest.approx(expected_weight)

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


