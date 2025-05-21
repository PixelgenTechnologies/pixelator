"""Webreport builder.

The WebreportBuilder is used to inject all webreport data into the
template and write the final webreport to a file.

Copyright © 2022 Pixelgen Technologies AB.
"""

import abc
import base64
import dataclasses
import gzip
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import lxml.etree
import semver
from lxml.etree import _Element as LxmlElement
from lxml.html import builder as E

from pixelator.common.report.json_encoder import PixelatorJSONEncoder

from .types import Metrics, QCReportData, SampleInfo

logger = logging.getLogger(__name__)

DEFAULT_QC_REPORT_TEMPLATE = Path(__file__).parent / "template.html"


@dataclasses.dataclass(frozen=True)
class DataInjectionPoint:
    """Data injection point in the template.

    This class is used to represent a data injection point in the template.
    Each injection point is a script tag with a specific data type and dataset id.
    """

    data_type: str
    content_type: str = "application/octet-stream;base64"
    accessor: Callable[[QCReportData], Any] | None = dataclasses.field(default=None)


class BaseQCReportBuilder(metaclass=abc.ABCMeta):
    """Base class for manipulating a QC report from a html template.

    This class is used to inject data into a QC report template to generate a complete report.
    It can also be used to extract data from a report using the `extract_field` method.

    Each custom script tag required by the template is represented by a `DataInjectionPoint`
    object in the `_DATA_FIELDS` class variable.

    Concrete child classes should implement the `write` method to inject the data into the template.
    """

    _DATA_FIELDS: ClassVar[list[DataInjectionPoint]] = []

    def __init__(
        self,
        template: Union[str, Path] = DEFAULT_QC_REPORT_TEMPLATE,
        version_constraints: Optional[List[str]] = None,
        json_options: Optional[Dict[str, Any]] = None,
    ):
        """Construct a QC report builder given a html template.

        :param template: path to the QC report template
        :param version_constraints: list of version constraints for the template
        :param json_options: options to pass to json.dumps when serializing data
        :raises FileNotFoundError: if the template file does not exist
        """
        if not Path(template).exists():
            raise FileNotFoundError(
                f"QC Report template not found at specified location: {template}"
            )
        self.template = Path(template)
        self._json_options = json_options or {"indent": None, "separators": (",", ":")}
        self._data_field_map = {field.data_type: field for field in self._DATA_FIELDS}
        self._version_constraints = version_constraints or []

    def _get_field(self, field_name: str) -> DataInjectionPoint | None:
        return self._data_field_map.get(field_name)

    def _load_template(self) -> Tuple[LxmlElement, LxmlElement]:
        """Load and parse the QC report template.

        :raises AssertionError: if no body tag is found in the template
        :raises AssertionError: if the QC report template version is not supported
        :rtype: Tuple[LxmlElement, LxmlElement]
        """
        logger.debug("Loading QC report template %s", self.template)

        # create a parser
        # Use huge_tree to allow large lxml objects and prevent
        # silent truncation of the html output.
        parser = lxml.etree.HTMLParser(huge_tree=True)

        # load and parse QC report template
        document = None
        with open(self.template, "rb") as f:
            document = lxml.html.parse(f, parser)

        root = document.getroot()
        body = document.find("body")

        if body is None:
            raise AssertionError("Could not find body element in QC report template")

        version = root.xpath('//meta[@name="application-name"]/@data-version')
        version = version[0] if version else None
        self._check_version_compatibility(version)

        logger.debug("QC report template %s loaded", self.template)
        return root, body

    def _check_version_compatibility(self, version: str):
        if version == "undefined":
            logger.warning("QC report version not found in template")
            return

        for constraint in self._version_constraints:
            check = semver.VersionInfo.parse(version).match(constraint)
            if not check:
                raise AssertionError(
                    f"Unsupported QC report version. Version {version} does "
                    f"not satisfy constraint: {constraint}"
                )

    @abstractmethod
    def write(  # noqa: DOC502
        self,
        fp: BinaryIO,
        sample_info: SampleInfo,
        data: QCReportData,
        metrics_definition_file: Optional[Path] = None,
    ) -> None:
        """Inject given data into the QC report and write the results to a stream.

        :param fp: binary stream to write the report to
        :param sample_info: Sample information
        :param metrics: Metrics for the sample
        :param data: Data for dynamic figures
        :param metrics_definition_file: Path to the metrics definition file
        :raises: AssertionError if not body tag is found in the template

        :example:
             root, body = self._load_template()

            # create the necessary script elements
            elements = []

            if metrics_definition_file is not None:
            # inject the elements in the body (in order for a more structured html)
            for idx, el in enumerate(elements):
                body.insert(idx, el)

            # write to html
            lxml.etree.ElementTree(root).write(fp, method="html", encoding="utf-8")

            logger.debug("Data injected in PNA QC report template %s", self.template)
        """
        raise NotImplementedError

    def _remove_existing_data_element(
        self, body: LxmlElement, field: str | DataInjectionPoint
    ) -> LxmlElement:
        if isinstance(field, str):
            field_desc = self._get_field(field)
            if field_desc is None:
                raise ValueError(f"Field {field} not found in data fields")
        else:
            field_desc = field

        res = body.cssselect(f'script[data-type="{field_desc.data_type}"]')
        if len(res) > 0:
            data_element = res[0]
            parent = data_element.getparent()
            if parent:
                parent.remove(data_element)

        return body

    def _build_elements(self, data: QCReportData):
        elements = {}
        for field in self._DATA_FIELDS:
            if field.accessor is None:
                continue
            else:
                data_field = field.accessor(data)
                if data_field is not None:
                    elements[field.data_type] = self._build_data_element(
                        field.data_type, data_field
                    )

        return elements

    def _build_data_element(self, data_type: str, data: str) -> LxmlElement:
        """Build a script element with the given data and data type."""
        data_descriptor = self._data_field_map[data_type]

        html_elem = E.SCRIPT(
            **{
                "type": data_descriptor.content_type,
                "data-type": data_descriptor.data_type,
                "data-dataset-id": "0",
            }
        )
        html_element_content = data
        if data_descriptor.content_type == "application/octet-stream;base64":
            html_element_content = self._compress_data(data)

        html_elem.text = html_element_content
        return html_elem

    @staticmethod
    def _compress_data(data: str):
        """Compress the data using gzip and encode with base64."""
        return base64.b64encode(gzip.compress(data.encode("utf-8")))

    def extract_field(self, body: LxmlElement, field_name: str) -> str:
        """Extract the data for a given field from the template.

        :param body: the body element of a parsed qc report
        :param field_name: the name of the field to extract
        :returns the data for the given field extracted from the report
        """
        field = self._get_field(field_name)
        if field is None:
            raise ValueError(f"Field {field_name} not found in data fields")

        selector = "script[data-type='{}']".format(field.data_type)
        element = body.cssselect(selector)
        if not element:
            raise ValueError(f"No data found for selector {selector}")

        data = element[0].text or ""
        if field.content_type == "application/octet-stream;base64":
            return gzip.decompress(base64.b64decode(data)).decode("utf-8")

        return data


class PNAQCReportBuilder(BaseQCReportBuilder):
    """Build a qc report from a html template and the required data (CSV and JSON).

    This will parse a qc report template and, using the write method, inject CSV
    and JSON strings into the template using `html <script>` tags under the
    `<body>` tag.

    Tags are labeled with the custom attributes `data-type` and `data-dataset-id`
    to identify the data.

    Note that since we are creating a single sample report (for now) the
    `data-dataset-id` is always set to '0'.

    >>> builder = PNAQCReportBuilder("./path/to/template.html")
    >>> with open("report.html", "wb") as fp:
    ...    builder.write(fp, sample_info, metrics, data)
    """

    VERSIONS_CONSTRAINTS: ClassVar[List[str]] = [">=0.1.0"]

    _DATA_FIELDS = [
        DataInjectionPoint(
            "metric-definitions",
            "application/octet-stream;base64",
        ),
        DataInjectionPoint("metrics", "application/octet-stream;base64"),
        DataInjectionPoint(
            "ranked-component-size",
            "application/octet-stream;base64",
            lambda data: data.ranked_component_size,
        ),
        DataInjectionPoint(
            "component-data",
            "application/octet-stream;base64",
            lambda data: data.component_data,
        ),
        DataInjectionPoint(
            "antibody-counts",
            "application/octet-stream;base64",
            lambda data: data.antibody_counts,
        ),
        DataInjectionPoint(
            "antibody-percentages",
            "application/octet-stream;base64",
            lambda data: data.antibody_percentages,
        ),
        DataInjectionPoint(
            "proximity-data",
            "application/octet-stream;base64",
            lambda data: data.proximity_data,
        ),
    ]

    def __init__(self, template: Union[str, Path] = DEFAULT_QC_REPORT_TEMPLATE):
        """Construct a QC report builder given a html template.

        :param template: path to the QC report template
        :raises FileNotFoundError: if the template file does not exist
        """
        super().__init__(template, self.VERSIONS_CONSTRAINTS)

    def write(  # noqa: DOC502
        self,
        fp: BinaryIO,
        sample_info: SampleInfo,
        data: QCReportData,
        metrics_definitions: Optional[Path | IO[str]] = None,
    ) -> None:
        """Inject given data into the QC report and write the results to a stream.

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

        if metrics_definitions is not None:
            elements.append(
                self._build_metric_definition_file_element(metrics_definitions, body)
            )

        elements.append(
            self._build_sample_and_metrics_element(sample_info, data.metrics)
        )

        default_elements = self._build_elements(data)
        drop_elements = {"metrics", "metric-definitions"}
        elements.extend(
            [v for k, v in default_elements.items() if k not in drop_elements]
        )

        # inject the elements in the body (in order for a more structured html)
        for idx, el in enumerate(elements):
            body.insert(idx, el)

        # write to html
        lxml.etree.ElementTree(root).write(fp, method="html", encoding="utf-8")

        logger.debug("Data injected in PNA QC report template %s", self.template)

    def _build_metric_definition_file_element(
        self, metrics_definitions: Path | IO[str], template_body: LxmlElement
    ) -> LxmlElement:
        """Create a lxml HTML object to inject the metrics definition file."""
        # Remove any existing default metric definitions already present in the template
        self._remove_existing_data_element(template_body, "metric-definitions")

        if isinstance(metrics_definitions, Path):
            with open(metrics_definitions, "r") as f:
                data = f.read()
        else:
            data = metrics_definitions.read()

        return self._build_data_element("metric-definitions", data)

    def _build_sample_and_metrics_element(
        self, sample_info: SampleInfo, metrics: Metrics
    ) -> LxmlElement:
        """Create a lxml HTML object to inject the metrics and sample info."""
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
        data = json.dumps(combined, **self._json_options, cls=PixelatorJSONEncoder)
        metrics_el.text = self._compress_data(data)
        return metrics_el
