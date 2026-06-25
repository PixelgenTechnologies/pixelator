Pixelator documentation
=======================

Pixelgen Technologies provides Pixelator, a suite of open source
software solutions that empower users working with
`Molecular Pixelation (MPX) <https://software.pixelgen.com/common/glossary/#mpx>`_
and `Proximity Network (PNA) <https://software.pixelgen.com/common/glossary/#pna>`_
assays in data processing and analysis.

Pixelator can be used in two ways: as a data processing pipeline
(nf-core/pixelator) and as a programming library (pixelator).
The pipeline nf-core/pixelator consists of several steps and will produce
ready-to-analyze outputs from your initial FASTQ sequencing libraries.
Usage of Pixelator as a programming library is covered in the API reference and in
our data analysis sections for MPX and PNA (see "Pixelgen software site" below).


.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Python API overview
      :link: api/overview
      :link-type: doc

      Main functions and classes for PNA analysis with the Pixelator Python API.

   .. grid-item-card:: Python API reference
      :link: api/index
      :link-type: doc

      Browse Python modules, classes, functions, and methods.

   .. grid-item-card:: Command-line interface
      :link: cli/index
      :link-type: doc

      Browse Pixelator command-line interface commands, options, and arguments.

   .. grid-item-card:: Pixelgen software site
      :link: https://software.pixelgen.com/
      :link-type: url

      See the Pixelgen software site for information about
      software, analysis, and datasets.

      .. .. grid-item-card:: Tutorials
      .. :link: https://software.pixelgen.com/pna-analysis/introduction/
      .. :link-type: url

      .. See tutorials and code examples for getting started
      .. with Proximity Network Assay (PNA) data analysis
      .. using Pixelator as a Python library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   api/index
   cli/index
