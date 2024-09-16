"""Graph --- :mod:`mdpath.src.graph`
==============================================================================

This module contains the class `GraphBuilder` which generates a graph of residues within a certain distance of each other.
Graph edges are assigned weights based on mutual information differences.
Paths between distant residues are calculated based on the shortest path with the highest total weight.

Classes
--------

:class:`GraphBuilder`
"""

import networkx as nx
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from Bio import PDB
from typing import Tuple, List
from mdpath.src.structure import StructureCalculations


class GraphBuilder:
    def __init__(
        self, pdb: str, last_residue: int, mi_diff_df: pd.DataFrame, graphdist: int
    ) -> None:
        self.pdb = pdb
        self.end = last_residue
        self.mi_diff_df = mi_diff_df
        self.dist = graphdist
        self.graph = self.graph_builder()

    def graph_skeleton(self) -> nx.Graph:
        """Generates a graph of residues within a certain distance of each other.

        Args:
            dist (float, optional): Cutoff distance for graph egdes. Distance of residues to eachother in Angström. Defaults to 5.0.

        Returns:
            residue_graph (nx.Graph): Graph of residues within a certain distance of each other.
        """
        residue_graph = nx.Graph()
        structure_calc = StructureCalculations(self.pdb)
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", self.pdb)
        heavy_atoms = ["C", "N", "O", "S"]
        residues = [
            res for res in structure.get_residues() if PDB.Polypeptide.is_aa(res)
        ]
        for res1, res2 in tqdm(
            combinations(residues, 2),
            desc="\033[1mBuilding residue graph\033[0m",
            total=len(residues) * (len(residues) - 1) // 2,
        ):
            res1_id = res1.get_id()[1]
            res2_id = res2.get_id()[1]
            if res1_id <= self.end and res2_id <= self.end:
                for atom1 in res1:
                    if atom1.element in heavy_atoms:
                        for atom2 in res2:
                            if atom2.element in heavy_atoms:
                                distance = structure_calc.calculate_distance(
                                    atom1.coord, atom2.coord
                                )
                                if distance <= self.dist:
                                    residue_graph.add_edge(
                                        res1.get_id()[1], res2.get_id()[1], weight=0
                                    )
        return residue_graph

    def graph_assign_weights(self, residue_graph: nx.Graph) -> nx.Graph:
        """Assignes edge weights to the graph based on mutual information differences.

        Args:
            residue_graph (nx.Graph): Base residue graph.
            mi_diff_df (pd.DataFrame): Pandas dataframe with mutual information differences.

        Returns:
            residue_graph (nx.Graph): Residue graph with edge weights assigned.
        """
        for edge in residue_graph.edges():
            u, v = edge
            pair = ("Res " + str(u), "Res " + str(v))
            if pair in self.mi_diff_df["Residue Pair"].apply(tuple).tolist():
                weight = self.mi_diff_df.loc[
                    self.mi_diff_df["Residue Pair"].apply(tuple) == pair,
                    "MI Difference",
                ].values[0]
                residue_graph.edges[edge]["weight"] = weight
        return residue_graph

    def graph_builder(self):
        graph = self.graph_skeleton()
        graph = self.graph_assign_weights(graph)
        return graph

    def max_weight_shortest_path(
        self, source: int, target: int
    ) -> Tuple[List[int], float]:
        """Finds the shortest path between 2 nodes with the highest total weight among all shortest paths.

        Args:
            graph (nx.Graph): Input graph.
            source (int): Starting node.
            target (int): Target node.

        Returns:
            best_path (List[int]): List of nodes in the shortest path with the highest weight.
            total_weight (float): Total weight of the shortest path.
        """
        all_shortest_paths = list(
            nx.all_shortest_paths(self.graph, source=source, target=target)
        )

        max_weight = -float("inf")
        best_path = None

        for path in all_shortest_paths:
            path_weight = sum(
                self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
            )
            if path_weight > max_weight:
                max_weight = path_weight
                best_path = path

        return best_path, max_weight

    def collect_path_total_weights(
        self, df_distant_residues: pd.DataFrame
    ) -> list[tuple[list[int], float]]:
        """Wrapper function to collect the shortest path and total weight between distant residues.

        Args:
            residue_graph (nx.Graph): Residue graph.
            df_distant_residues (pd.DataFrame): Panda dataframe with distant residues.

        Returns:
            path_total_weights (list[tuple[list[int], float]]): List of tuples with the shortest path and total weight between distant residues.
        """
        path_total_weights = []
        for index, row in df_distant_residues.iterrows():
            try:
                shortest_path, total_weight = self.max_weight_shortest_path(
                    row["Residue1"], row["Residue2"]
                )
                path_total_weights.append((shortest_path, total_weight))
            except nx.NetworkXNoPath:
                continue
        return path_total_weights
