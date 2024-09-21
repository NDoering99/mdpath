Running MDPath
==============

This page details the primary functionality of **MDPath**, which is the analysis of signaling pathways in proteins based on molecular dynamics simulations.

**Usage**

To use the main function of **MDPath**, always use the prefix `mdpath` followed by the corresponding mandatory and optional flags. 

**Mandatory Inputs**

The following inputs are required when running **MDPath** from the command line:

.. code-block:: text

    -top    Topology file of your molecular dynamics simulation [.pdb]
    -traj   Trajectory file of your molecular dynamics simulation [.dcd]

**Optional Inputs**
In addition to the mandatory inputs, the following optional parameters can be provided to customize the analysis:

.. code-block:: text

    -cpu       Number of virtual cores to use [n]
    -lig       Important residues a path must cross in order to be plotted [list of residues, e.g.: 1 2 3 4 ...]
    -bs        Number of bootstrap samples to be calculated for your system [n]
    -fardist   Minimum distance in angstroms for residues to be considered as start and end points for pathways [n]
    -closedist Minimum distance in angstroms for residues to be considered overlapping for clustering [n]
    -graphdist Minimum distance in angstroms for residues to be considered to have corresponding movements for graph building [n]
    -numpath   Number of top paths with the highest normalized mutual information to be plotted [n]

An example command line input for running **MDPath** could look like this:

.. code-block:: text

    mdpath -top [Path to your topology].pdb -traj [Path to your trajectory].dcd
