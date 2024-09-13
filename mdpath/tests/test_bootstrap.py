import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch
import pickle
import os
from mdpath.src.bootstrap import BootstrapAnalysis


@pytest.fixture
def load_pickle_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bs = os.path.join(current_dir, "bs_init.pkl")
    with open(bs, "rb") as file:
        df_all_residues = pickle.load(file)
        df_distant_residues = pickle.load(file)
        sorted_paths = pickle.load(file)
        num_bootstrap_samples = pickle.load(file)
        numpath = pickle.load(file)
        last_residue = pickle.load(file)
        graphdist = pickle.load(file)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb = os.path.join(current_dir, "first_frame.pdb")

    return (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    )


def test_bootstrap_analysis_init(load_pickle_data):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    assert_frame_equal(analysis.df_all_residues, df_all_residues)
    assert_frame_equal(analysis.df_distant_residues, df_distant_residues)

    assert analysis.sorted_paths == sorted_paths
    assert analysis.num_bootstrap_samples == num_bootstrap_samples
    assert analysis.numpath == numpath
    assert analysis.pdb == pdb
    assert analysis.last_residue == last_residue
    assert analysis.graphdist == graphdist
    assert analysis.num_bins == 35


@pytest.fixture
def load_dataframe():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mi_df = os.path.join(current_dir, "mi_diff_df.csv")
    df_dihedral = pd.read_csv(mi_df)
    return df_dihedral


def test_create_bootstrap_sample(load_dataframe, load_pickle_data):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    bootstrap_sample = analysis.create_bootstrap_sample(load_dataframe)
    assert bootstrap_sample.shape == load_dataframe.shape
    assert list(bootstrap_sample.columns) == list(load_dataframe.columns)
    assert not bootstrap_sample.equals(
        load_dataframe
    ), "Bootstrap sample should not be identical to the original data"
