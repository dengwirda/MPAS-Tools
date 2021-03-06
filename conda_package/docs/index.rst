MPAS-Tools
==========

.. image:: images/so60to10.png
   :width: 500 px
   :align: center

MPAS-Tools includes a python package, compiled Fortran, C and C++ tools, and
scripts for supporting initialization, visualization and analysis of Model for
Prediction Across Scales (MPAS) components.  These tools are used by the
`COMPASS <https://github.com/MPAS-Dev/MPAS-Model/tree/master/testing_and_setup/compass>`_
(Configuring of MPAS Setups) framework within
`MPAS-Model <https://github.com/MPAS-Dev/MPAS-Model>`_ used to create
ocean and land-ice test cases,
the `MPAS-Analysis <https://github.com/MPAS-Dev/MPAS-Analysis>`_ package for
analyzing simulations, and in other MPAS-related workflows.

User's Guide
============

.. toctree::
   :maxdepth: 2

   mesh_creation
   mesh_conversion
   interpolation

   cime

   visualization

Ocean Tools
-----------

.. toctree::
   :maxdepth: 2

   ocean/mesh_creation
   ocean/coastal_tools
   ocean/coastline_alteration
   ocean/moc

Developer's Guide
=================
.. toctree::
   :maxdepth: 2

   making_changes
   testing_changes
   building_docs

   api

Indices and tables
==================

* :ref:`genindex`

Authors
=======
.. toctree::
   :maxdepth: 1

   authors

Versions
========
.. toctree::
   :maxdepth: 1

   versions
