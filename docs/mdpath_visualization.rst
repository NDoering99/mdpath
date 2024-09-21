MDPath visualization
====================

Currently, two options for 3D visualization are integrated into MDPath.

**NGL Visualization**
==============
For visualization using NGLView, open the corresponding Jupyter notebook. Then, insert the path to your topology or first_frame.pdb, along with the path to your desired MDPath output, such as quick_precomputed_clusters_paths.json or precomputed_clusters_paths.json.


.. figure:: /_static/images/ngl_script.png
   :figwidth: 725px
   :align: center


Afterward, simply execute the notebook and navigate through the visualization as you would in NGLView.

.. figure:: /_static/images/ngl_example.png
   :figwidth: 725px
   :align: center


**Pymol Visualization**
==============
To visualize the paths in PyMOL, start by launching PyMOL according to your installation method.
Under the "File" menu, select the option "Run Script..." and navigate to the mdpath folder. 


.. figure:: /_static/images/pymol_run_script.png
   :figwidth: 725px
   :align: center



From there, select the vis_pymol.py script.
Now, you can run the script from the command line using the following command:

.. figure:: /_static/images/python_cmd.png
   :figwidth: 725px
   :align: center


Afterward, you can execute the visualization steps in PyMOL as you normally would.



.. figure:: /_static/images/pymol_ray_example.png
   :figwidth: 725px
   :align: center
 
