Python API overview
====================

Primary entry points for PNA analysis. See the
`tutorials <https://software.pixelgen.com/pna-analysis/python/tutorials/introduction/>`_
for usage examples.

**Reader**

* :func:`pixelator.read_pna`


**Datasets**

* :class:`pixelator.pna.pixeldataset.download.DownloadableDatasets`

**PNAPixelDataset**

* :class:`pixelator.pna.pixeldataset.PNAPixelDataset`

**Plotting**

* :func:`pixelator.pna.plot.molecule_rank_plot`

**Abundance normalization**

* :func:`pixelator.common.statistics.clr_transformation`
* :func:`pixelator.common.statistics.dsb_normalize`

**Analysis**

* :func:`pixelator.pna.analysis.calculate_differential_proximity`
