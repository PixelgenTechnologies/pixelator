"""
The WebreportBuilder is used to inject all webreport data into the
template and write the final webreport to a file.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import base64
import dataclasses
import gzip
import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Dict, List, Optional, Tuple, Union

import lxml.etree
import semver
from lxml.etree import _Element as LxmlElement
from lxml.html import builder as E

from pixelator.report.webreport.types import Metrics, SampleInfo, WebreportData
from pixelator.types import PathType

logger = logging.getLogger(__name__)

DEFAULT_WEBREPORT_TEMPLATE = Path(__file__).parent / "template.html"


class WebreportBuilder:
    """
    Build a webreport from a html template and the required data (CSV and JSON).

    This will parse a webreport template and, using the write method, inject CSV
    and JSON strings into the template using `html <script>` tags under the
    `<body>` tag.

    Tags are labeled with the custom attributes `data-type` and `data-dataset-id`
    to identify the data.

    Note that since we are creating a single sample report (for now) the
    `data-dataset-id` is always set to '0'.

    >>> builder = WebreportBuilder("./path/to/template.html")
    >>> with open("report.html", "wb") as fp:
    ...    builder.write(fp, sample_info, metrics, data)
    """

    _JSON_OPTIONS: ClassVar[Dict[str, Any]] = {"indent": None, "separators": (",", ":")}
    VERSIONS_CONSTRAINTS: ClassVar[List[str]] = ["<0.7.0", ">=0.5.0"]

    def __init__(self, template: Union[str, Path] = DEFAULT_WEBREPORT_TEMPLATE):
        """
        Construct a webreport builder given a html template.

        :param template: path to the webreport template
        :raises FileNotFoundError: if the template file does not exist
        """
        if not Path(template).exists():
            raise FileNotFoundError(
                f"Webreport template not found at specified location: {template}"
            )
        self.template = Path(template)

    def _load_template(self) -> Tuple[LxmlElement, LxmlElement]:
        """
        Load and parse the webreport template.

        :raises AssertionError: if no body tag is found in the template
        :raises AssertionError: if the webreport template version is not supported
        """
        logger.debug("Loading web report template %s", self.template)

        # create a parser
        # Use huge_tree to allow large lxml objects and prevent
        # silent truncation of the html output.
        parser = lxml.etree.HTMLParser(huge_tree=True)

        # load and parse webreport template
        document = None
        with open(self.template, "rb") as f:
            document = lxml.html.parse(f, parser)

        root = document.getroot()
        body = document.find("body")

        if body is None:
            raise AssertionError("Could not find body element in webreport template")

        version = root.xpath('//meta[@name="application-name"]/@data-version')
        version = version[0] if version else None
        self._check_version_compatibility(version)

        logger.debug("Web report template %s loaded", self.template)
        return root, body

    def _check_version_compatibility(self, version: str):
        for constraint in self.VERSIONS_CONSTRAINTS:
            check = semver.VersionInfo.parse(version).match(constraint)
            if not check:
                raise AssertionError(
                    f"Unsupported webreport version. Version {version} does "
                    f"not satisfy constraint: {constraint}"
                )

    def write(
        self,
        fp: BinaryIO,
        sample_info: SampleInfo,
        metrics: Metrics,
        data: WebreportData,
        metrics_definition_file: Optional[PathType] = None,
    ) -> None:
        """
        Inject given data into the webreport and write the results to a stream.

        :param fp: binary stream to write the report to
        :param sample_info: Sample information
        :param metrics: Metrics for the sample
        :param data: Data for dynamic figures
        :param metrics_definition_file: Path to the metrics definition file
        :raises: AssertionError if not body tag is found in the template
        """
        root, body = self._load_template()

        # create the necessary script elements
        elements = []

        if metrics_definition_file is not None:
            elements.append(
                self._build_metric_definition_file_element(
                    metrics_definition_file, body
                )
            )

        elements.append(self._build_sample_and_metrics_element(sample_info, metrics))

        if data.ranked_component_size is not None:
            elements.append(
                self._build_ranked_component_size_element(data.ranked_component_size)
            )

        if data.component_data is not None:
            elements.append(self._build_component_data_element(data.component_data))

        if data.antibodies_per_cell is not None:
            elements.append(
                self._build_antibodies_per_component_element(data.antibodies_per_cell)
            )

        if data.sequencing_saturation is not None:
            elements.append(
                self._build_sequencing_saturation_element(data.sequencing_saturation)
            )

        if data.antibody_percentages is not None:
            elements.append(
                self._build_antibody_percentages_element(data.antibody_percentages)
            )

        if data.antibody_counts is not None:
            elements.append(self._build_antibody_counts_element(data.antibody_counts))

        # inject the elements in the body (in order for a more structured html)
        for idx, el in enumerate(elements):
            body.insert(idx, el)

        # write to html
        lxml.etree.ElementTree(root).write(fp, method="html", encoding="utf-8")

        logger.debug("Data injected in web report template %s", self.template)

    def _build_metric_definition_file_element(
        self, metrics_definition_file: PathType, template_body: LxmlElement
    ) -> LxmlElement:
        """
        Create a lxml HTML object to inject the metrics definition file.
        """
        template_body.cssselect('script[data-type="metric-definitions"]')
        if len(template_body) > 0:
            metrics_definition_file_el = template_body[0]
            parent = metrics_definition_file_el.getparent()
            if parent:
                parent.remove(metrics_definition_file_el)

        metrics_definition_file_el = E.SCRIPT(
            **{
                "type": "application/octet-stream;base64",
                "data-type": "metric-definitions",
            }
        )

        with open(metrics_definition_file, "r") as f:
            data = f.read()

        metrics_definition_file_el.text = self._compress_data(data)
        return metrics_definition_file_el

    def _build_sample_and_metrics_element(
        self, sample_info: SampleInfo, metrics: Metrics
    ) -> LxmlElement:
        """
        Create a lxml HTML object to inject the metrics and sample info.
        """
        metrics_el = E.SCRIPT(
            **{
                "type": "application/octet-stream;base64",
                "data-type": "metrics",
                "data-dataset-id": "0",
            }
        )

        combined = {
            "info": dataclasses.asdict(sample_info),
            "metrics": metrics,
        }
        data = json.dumps(combined, **self._JSON_OPTIONS)
        metrics_el.text = self._compress_data(data)
        return metrics_el

    def _build_ranked_component_size_element(self, data: str) -> LxmlElement:
        """
        Create a lxml HTML injecting the ranked component data.

        This data is used for the rank plot and the component size vs marker
        scatter plot.
        """
        ranked_component_size_el = E.SCRIPT(
            **{
                "type": "application/octet-stream;base64",
                "data-type": "ranked-component-size",
                "data-dataset-id": "0",
            }
        )
        ranked_component_size_el.text = self._compress_data(data)
        return ranked_component_size_el

    def _build_component_data_element(self, data: str) -> LxmlElement:
        """
        Create a lxml HTML injecting the component data.
        """
        component_data_el = E.SCRIPT(
            **{
                "type": "application/octet-stream;base64",
                "data-type": "component-data",
                "data-dataset-id": "0",
            }
        )
        component_data_el.text = self._compress_data(data)
        return component_data_el

    def _build_antibodies_per_component_element(self, data: str) -> LxmlElement:
        """
        Create a lxml HTML injecting the antibodies_per_cell data.
        """
        antibodies_per_cell_el = E.SCRIPT(
            **{
                "type": "text/csv",
                "data-type": "antibodies-per-cell",
                "data-dataset-id": "0",
            }
        )
        antibodies_per_cell_el.text = data
        return antibodies_per_cell_el

    def _build_sequencing_saturation_element(self, data: str):
        """
        Create an HTML object injecting the sequencing saturation data.
        """
        sequencing_saturation_el = E.SCRIPT(
            **{
                "type": "text/csv",
                "data-type": "sequencing-saturation",
                "data-dataset-id": "0",
            }
        )
        sequencing_saturation_el.text = data
        return sequencing_saturation_el

    def _build_antibody_percentages_element(self, data: str) -> LxmlElement:
        """
        Create a HTML object injecting the antibody counts data.
        """
        antibody_counts_el = E.SCRIPT(
            **{
                "type": "text/csv",
                "data-type": "antibody-counts",
                "data-dataset-id": "0",
            }
        )
        antibody_counts_el.text = data
        return antibody_counts_el

    def _build_antibody_counts_element(self, data: str) -> LxmlElement:
        """
        Create an HTML object injecting the antibody counts data.
        """
        antibody_distribution_el = E.SCRIPT(
            **{
                "type": "application/octet-stream;base64",
                "data-type": "antibody-distribution",
                "data-dataset-id": "0",
            }
        )
        antibody_distribution_el.text = self._compress_data(data)
        return antibody_distribution_el

    @staticmethod
    def _compress_data(data: str):
        """
        Compress the data using gzip and encode with base64.
        """
        return base64.b64encode(gzip.compress(data.encode("utf-8")))
