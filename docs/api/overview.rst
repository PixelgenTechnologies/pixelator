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
* :meth:`pixelator.pna.pixeldataset.PNAPixelDataset.layouts`
* :class:`pixelator.pna.pixeldataset.layouts.Layouts`

  :meth:`~pixelator.pna.pixeldataset.PNAPixelDataset.precomputed_layouts` and
  :class:`~pixelator.pna.pixeldataset.precomputed_layouts.PreComputedLayouts`
  are deprecated; use :meth:`~pixelator.pna.pixeldataset.PNAPixelDataset.layouts`
  instead.

**Plotting**

* :func:`pixelator.pna.plot.molecule_rank_plot`
* :func:`pixelator.pna.plot.proximity_heatmap`

**Abundance normalization**

* :func:`pixelator.common.statistics.clr_transformation`
* :func:`pixelator.common.statistics.dsb_normalize`

**Analysis**

* :func:`pixelator.pna.analysis.calculate_differential_proximity`
* :func:`pixelator.pna.analysis.summarize_proximity_scores`
