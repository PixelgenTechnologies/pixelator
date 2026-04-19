"""Module for utilities for working with pixeldatasets.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from anndata import AnnData, ImplicitModificationWarning

logger = logging.getLogger(__name__)


def update_metrics_anndata(adata: AnnData, inplace: bool = True) -> Optional[AnnData]:
    """Update any metrics in the AnnData instance.

    This will  update the QC metrics (`var` and `obs`) of
    the AnnData object given as input. This function is typically used
    when the AnnData object has been filtered and one wants the QC metrics
    to be updated accordingly.

    :param adata: an AnnData object
    :param inplace: If `True` performs the operation inplace
    :returns: the updated AnnData object or None if inplace is True
    :rtype: Optional[AnnData]
    """
    logger.debug(
        "Updating metrics in AnnData object with %i components and %i markers",
        adata.n_obs,
        adata.n_vars,
    )

    if not inplace:
        adata = adata.copy()

    df = adata.to_df()

    # update the var layer (antibody metrics)
    # we ignore the warning here, since we actually want to force and update of the
    # `adata.var` data frame.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        adata.var["antibody_count"] = df.sum().astype(int)
    adata.var["components"] = (df != 0).sum()
    adata.var["antibody_pct"] = (
        adata.var["antibody_count"] / adata.var["antibody_count"].sum()
    ).astype(np.float32)

    # update the obs layer (components metrics)
    adata.obs["n_antibodies"] = np.sum(adata.X > 0, axis=1, dtype=np.uint32)

    logger.debug("Metrics in AnnData object updated")
    return None if inplace else adata
