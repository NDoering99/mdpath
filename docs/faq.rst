Frequently Asked Questions
==========================

What OS does MDPath support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MDPath has been evaluated to run on Windows, macOS, and Linux distributions for Python versions >3.10.

Can MDPath run on older Python versions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MDPath has been successfully tested on Python versions as old as 3.9. However, we do not recommend using older distributions, as they are not consistently tested.

What should I keep in mind when setting up simulations to be evaluated by MDPath?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensure that the state you aim to capture is consistent throughout the simulation, as major conformational shifts within a single analysis can lead to problematic outcomes. In our tests, we saved 1,000 frames from simulations run for 200 ns each. For the complete setup, please refer to our main paper.

What do I need to cite when using MDPath in my work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please cite the main MDPath publication and any relevant papers mentioned in the documentation. Refer to the “Citing MDPath” section for the full citation details.

Do you plan to add other visualization options?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, we are continuously working on expanding visualization features. Stay tuned for updates in future releases.

How can I implement this for a system not related to G-protein coupled receptors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MDPath can be adapted for other systems by modifying the input files and parameters according to your specific needs. Refer to the documentation for guidelines on customizing the analysis for different molecular systems. Once we have validated papers describing other systems, we will share and update the tool’s parameters accordingly to ensure adherence to good scientific practices.

What does the bootstrap flag do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The bootstrap flag enables the bootstrapping method to assess the stability and reliability of the results. It involves resampling the data to estimate variability and confidence intervals, providing a measure of the robustness of the analysis.

Keep in mind that while a low standard error or high confidence intervals indicate statistical validity within the scope of the analysis, they do not guarantee that the observed findings are biologically correct.

What does the -lig flag do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The -lig flag, named from its initial use in analyzing protein-ligand interactions, filters the paths to include only those that pass through a predefined residue. This enables the analysis of how protein-ligand interactions affect the protein's conformational state. Additionally, it can be used to explore protein-protein interactions, highlight specific paths in an allosteric network, or investigate the effects of point mutations. The possibilities are extensive, limited only by your computational resources and creativity.

How can I contribute to the project? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As an open-source project, everyone is invited to contribute. You can:

1. Create Pull Requests: Submit a pull request with your improvements or new features. Each pull request will be reviewed before merging into the main branch.
2. Report Issues: Report bugs or suggest enhancements. Issue reporting is crucial for improving the tool.

Your contributions help enhance the project and are greatly appreciated!

What tools can I use to track dynamic protein-ligand interactions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common tools for tracking protein-ligand interactions include Dynophores, OpenMMDL, and Prolif.
