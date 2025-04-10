"""Copyright © 2025 Pixelgen Technologies AB."""

# """Test QC report generation.
#
# Copyright © 2022 Pixelgen Technologies AB.
# """

import dataclasses
import datetime
import io
import json

import lxml

from pixelator.pna.report.qcreport import PNAQCReportBuilder, QCReportData, SampleInfo


def test_report_builder(tmp_path):
    """Test the presence of the script tags with the correct data in the HTML report."""
    sample_name = "PNA055_Sample07_filtered_S7"
    builder = PNAQCReportBuilder()
    report_path = tmp_path / f"{sample_name}.qc-report.html"

    # report_data = collect_report_data(reporting, sample_id)
    qc_report_data = QCReportData(
        metrics={
            "number_of_reads": 99812,
            "reads_discarded_amplicon": 2405,
            "fraction_of_reads_discarded_amplicon": 0.02409529916242536,
            "reads_discarded_demux": 20,
            "fraction_of_reads_discarded_demux": 0.0002053240526861519,
            "node_saturation": 1.0,
            "fraction_q30_bases_in_pid1": 0.978598047368259,
            "fraction_q30_bases_in_pid2": 0.976195755951831,
            "fraction_q30_bases_in_umi1": 0.9697718996434694,
            "fraction_q30_bases_in_umi2": 0.9580841945944043,
            "fraction_q30_bases_in_lbs1": 0.97016165067683,
            "fraction_q30_bases_in_lbs2": 0.978912115748876,
            "fraction_q30_bases_in_uei": 0.9399345016271932,
            "fraction_q30_bases": 0.9687832501504746,
            "number_of_cells": 3,
            "median_reads_per_cell": 397,
            "median_markers_per_cell": 6,
            "median_average_k_coreness": 1.2222222222222223,
            "fraction_of_discarded_cells": 0.999870505460353,
            "spatial_coherence": 0.021432564092297022,
            "fraction_of_outlier_cells": 0.3333333333333333,
        },
        ranked_component_size="rank,size,selected\n1.0,288,True\n2.0,279,True\n3.0,237,True\n4.0,230,False\n5.0,224,False\n6.0,202,False\n7.0,197,False\n8.0,176,False\n9.0,166,False\n11.0,162,False\n12.0,158,False\n13.0,145,False\n14.0,141,False\n15.0,140,False\n16.0,135,False\n17.0,131,False\n18.0,125,False\n19.0,118,False\n20.0,113,False\n21.0,111,False\n22.0,110,False\n23.5,109,False\n25.0,107,False\n26.0,105,False\n27.0,97,False\n30.0,93,False\n31.5,92,False\n33.5,91,False\n35.0,90,False\n36.0,89,False\n38.5,84,False\n40.0,83,False\n42.0,81,False\n45.0,79,False\n46.0,77,False\n47.5,76,False\n49.5,75,False\n51.5,73,False\n53.0,71,False\n57.5,68,False\n59.5,67,False\n64.5,65,False\n66.5,64,False\n69.0,62,False\n72.0,61,False\n74.5,60,False\n76.0,59,False\n77.0,58,False\n79.0,57,False\n84.0,55,False\n87.0,54,False\n89.0,53,False\n93.5,52,False\n99.0,51,False\n102.0,50,False\n106.5,49,False\n111.5,48,False\n116.0,47,False\n121.0,46,False\n124.0,45,False\n126.5,44,False\n130.5,43,False\n135.5,42,False\n141.5,41,False\n149.0,40,False\n155.0,39,False\n158.5,38,False\n165.0,37,False\n174.0,36,False\n183.5,35,False\n191.5,34,False\n204.5,32,False\n212.0,31,False\n222.5,30,False\n233.5,29,False\n245.0,28,False\n276.0,26,False\n289.0,25,False\n305.5,24,False\n327.0,23,False\n381.5,21,False\n402.5,20,False\n430.5,19,False\n468.5,18,False\n511.5,17,False\n603.5,15,False\n655.5,14,False\n727.5,13,False\n833.0,12,False\n957.5,11,False\n1111.5,10,False\n1321.0,9,False\n1602.5,8,False\n2004.0,7,False\n2605.5,6,False\n3572.0,5,False\n5244.0,4,False\n8552.5,3,False\n16979.5,2,False\n",
        component_data="component,umap1,umap2,cluster,reads_in_component,n_antibodies,n_umi,n_edges\n099e0e79520b5df9,-3.8431678,-0.9213031,,447,7,279,293\nc3cbbdc4dcc53253,-2.350471,-1.0423394,,328,6,237,246\nd64c360022eec58f,-2.9546297,0.35776874,,397,5,288,301\n",
        antibody_percentages="antibody,count,percentage\nHLA-ABC,714,0.8880597\nB2M,0,0.0\nCD11b,0,0.0\nCD11c,0,0.0\nCD18,1,0.0012437811\nCD82,0,0.0\nCD3e,33,0.041044775\nCD4,5,0.0062189056\nCD8,1,0.0012437811\nTCRab,2,0.0024875621\nHLA-DR,0,0.0\nCD45,42,0.052238807\nCD14,0,0.0\nCD16,0,0.0\nCD19,2,0.0024875621\nmIgG1,0,0.0\nCD11a,1,0.0012437811\nCD45RB,0,0.0\nACTB,3,0.0037313432\n",
        antibody_counts="component,HLA-ABC,B2M,CD11b,CD11c,CD18,CD82,CD3e,CD4,CD8,TCRab,HLA-DR,CD45,CD14,CD16,CD19,mIgG1,CD11a,CD45RB,ACTB\n099e0e79520b5df9,223,0,0,0,0,0,27,0,1,1,0,24,0,0,0,0,1,0,2\nc3cbbdc4dcc53253,227,0,0,0,1,0,2,0,0,0,0,4,0,0,2,0,0,0,1\nd64c360022eec58f,264,0,0,0,0,0,4,5,0,1,0,14,0,0,0,0,0,0,0\n",
        proximity_data="""{"markers": ["ACTB", "CD11a", "CD18", "CD19", "CD3e", "CD4", "CD45", "CD8", "HLA-ABC", "TCRab", "mIgG1"], "join_count_z": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1931235431235431, 0.0, 0.21192886867891322, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0625, 0.0, -0.37209302325581395, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4640472860717734, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7292495292293485, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.6348537310952166, 0.0, 0.03473500211251339, 0.0, -0.6001492706947295, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4408307210031348, 0.0, -0.19165708762366476, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3706990009921611, -0.3125, -0.6771465564913955, -0.2727272727272727, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2223500629714145, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8879487665649706, 0.10740355766687243, -0.37886983562648324], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "log2_ratio": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2386771887821483, 0.0, -0.16686818794607042, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.08746284125033942, 0.0, -0.4563782946597432, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6440501599927262, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9135542624944075, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.3924697494158373, 0.0, -0.08935259308703218, 0.0, -0.36200559427284795, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5195196782284426, 0.0, -0.38665249307649807, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.39474438269881157, -0.39231742277876036, -0.33562942750463615, -0.34792330342030686, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20090730426553627, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04999551707372642, -0.19161613901918262, -0.608610134909639], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}""",
    )
    generation_time = datetime.datetime(2020, 1, 1, 0, 0, 0)

    info = SampleInfo(
        pixelator_version="1.0.0",
        generation_date=generation_time.isoformat(),
        sample_id=sample_name,
        sample_description="just a test",
        panel_name="human-sc-immunology-spatial-proteomics",
        panel_version="1.0.0",
        technology="PNA",
        technology_version="2.5",
        parameters=[],
    )

    with open(report_path, "wb") as f:
        builder.write(f, sample_info=info, data=qc_report_data)

    assert report_path.exists()

    parser = lxml.etree.HTMLParser(huge_tree=True)
    with open(report_path, "rb") as f:
        document = lxml.html.parse(f, parser)
        body = document.find("body")

    metrics_data = builder.extract_field(body, "metrics")
    assert json.loads(metrics_data) == {
        "info": dataclasses.asdict(info),
        "metrics": qc_report_data.metrics,
    }

    ranked_component_size_data = builder.extract_field(body, "ranked-component-size")
    assert ranked_component_size_data == qc_report_data.ranked_component_size

    component_data = builder.extract_field(body, "component-data")
    assert component_data == qc_report_data.component_data

    antibody_percentages = builder.extract_field(body, "antibody-percentages")
    assert antibody_percentages == qc_report_data.antibody_percentages

    antibody_counts = builder.extract_field(body, "antibody-counts")
    assert antibody_counts == qc_report_data.antibody_counts

    proximity_data = builder.extract_field(body, "proximity-data")
    assert proximity_data == qc_report_data.proximity_data


def test_report_builder_custom_definitions(tmp_path):
    sample_name = "PNA055_Sample07_filtered_S7"
    builder = PNAQCReportBuilder()
    report_path = tmp_path / f"{sample_name}.qc-report.html"

    # report_data = collect_report_data(reporting, sample_id)
    qc_report_data = QCReportData(
        metrics={
            "number_of_reads": 99812,
            "reads_discarded_amplicon": 2405,
            "fraction_of_reads_discarded_amplicon": 0.02409529916242536,
            "reads_discarded_demux": 20,
            "fraction_of_reads_discarded_demux": 0.0002053240526861519,
            "node_saturation": 1.0,
            "fraction_q30_bases_in_pid1": 0.978598047368259,
            "fraction_q30_bases_in_pid2": 0.976195755951831,
            "fraction_q30_bases_in_umi1": 0.9697718996434694,
            "fraction_q30_bases_in_umi2": 0.9580841945944043,
            "fraction_q30_bases_in_lbs1": 0.97016165067683,
            "fraction_q30_bases_in_lbs2": 0.978912115748876,
            "fraction_q30_bases_in_uei": 0.9399345016271932,
            "fraction_q30_bases": 0.9687832501504746,
            "number_of_cells": 3,
            "median_reads_per_cell": 397,
            "median_markers_per_cell": 6,
            "median_average_k_coreness": 1.2222222222222223,
            "fraction_of_discarded_cells": 0.999870505460353,
            "spatial_coherence": 0.021432564092297022,
            "fraction_of_outlier_cells": 0.3333333333333333,
        },
        ranked_component_size="rank,size,selected\n1.0,288,True\n2.0,279,True\n3.0,237,True\n4.0,230,False\n5.0,224,False\n6.0,202,False\n7.0,197,False\n8.0,176,False\n9.0,166,False\n11.0,162,False\n12.0,158,False\n13.0,145,False\n14.0,141,False\n15.0,140,False\n16.0,135,False\n17.0,131,False\n18.0,125,False\n19.0,118,False\n20.0,113,False\n21.0,111,False\n22.0,110,False\n23.5,109,False\n25.0,107,False\n26.0,105,False\n27.0,97,False\n30.0,93,False\n31.5,92,False\n33.5,91,False\n35.0,90,False\n36.0,89,False\n38.5,84,False\n40.0,83,False\n42.0,81,False\n45.0,79,False\n46.0,77,False\n47.5,76,False\n49.5,75,False\n51.5,73,False\n53.0,71,False\n57.5,68,False\n59.5,67,False\n64.5,65,False\n66.5,64,False\n69.0,62,False\n72.0,61,False\n74.5,60,False\n76.0,59,False\n77.0,58,False\n79.0,57,False\n84.0,55,False\n87.0,54,False\n89.0,53,False\n93.5,52,False\n99.0,51,False\n102.0,50,False\n106.5,49,False\n111.5,48,False\n116.0,47,False\n121.0,46,False\n124.0,45,False\n126.5,44,False\n130.5,43,False\n135.5,42,False\n141.5,41,False\n149.0,40,False\n155.0,39,False\n158.5,38,False\n165.0,37,False\n174.0,36,False\n183.5,35,False\n191.5,34,False\n204.5,32,False\n212.0,31,False\n222.5,30,False\n233.5,29,False\n245.0,28,False\n276.0,26,False\n289.0,25,False\n305.5,24,False\n327.0,23,False\n381.5,21,False\n402.5,20,False\n430.5,19,False\n468.5,18,False\n511.5,17,False\n603.5,15,False\n655.5,14,False\n727.5,13,False\n833.0,12,False\n957.5,11,False\n1111.5,10,False\n1321.0,9,False\n1602.5,8,False\n2004.0,7,False\n2605.5,6,False\n3572.0,5,False\n5244.0,4,False\n8552.5,3,False\n16979.5,2,False\n",
        component_data="component,umap1,umap2,cluster,reads_in_component,n_antibodies,n_umi,n_edges\n099e0e79520b5df9,-3.8431678,-0.9213031,,447,7,279,293\nc3cbbdc4dcc53253,-2.350471,-1.0423394,,328,6,237,246\nd64c360022eec58f,-2.9546297,0.35776874,,397,5,288,301\n",
        antibody_percentages="antibody,count,percentage\nHLA-ABC,714,0.8880597\nB2M,0,0.0\nCD11b,0,0.0\nCD11c,0,0.0\nCD18,1,0.0012437811\nCD82,0,0.0\nCD3e,33,0.041044775\nCD4,5,0.0062189056\nCD8,1,0.0012437811\nTCRab,2,0.0024875621\nHLA-DR,0,0.0\nCD45,42,0.052238807\nCD14,0,0.0\nCD16,0,0.0\nCD19,2,0.0024875621\nmIgG1,0,0.0\nCD11a,1,0.0012437811\nCD45RB,0,0.0\nACTB,3,0.0037313432\n",
        antibody_counts="component,HLA-ABC,B2M,CD11b,CD11c,CD18,CD82,CD3e,CD4,CD8,TCRab,HLA-DR,CD45,CD14,CD16,CD19,mIgG1,CD11a,CD45RB,ACTB\n099e0e79520b5df9,223,0,0,0,0,0,27,0,1,1,0,24,0,0,0,0,1,0,2\nc3cbbdc4dcc53253,227,0,0,0,1,0,2,0,0,0,0,4,0,0,2,0,0,0,1\nd64c360022eec58f,264,0,0,0,0,0,4,5,0,1,0,14,0,0,0,0,0,0,0\n",
        proximity_data="""{"markers": ["ACTB", "CD11a", "CD18", "CD19", "CD3e", "CD4", "CD45", "CD8", "HLA-ABC", "TCRab", "mIgG1"], "join_count_z": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1931235431235431, 0.0, 0.21192886867891322, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0625, 0.0, -0.37209302325581395, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4640472860717734, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7292495292293485, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.6348537310952166, 0.0, 0.03473500211251339, 0.0, -0.6001492706947295, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4408307210031348, 0.0, -0.19165708762366476, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3706990009921611, -0.3125, -0.6771465564913955, -0.2727272727272727, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2223500629714145, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8879487665649706, 0.10740355766687243, -0.37886983562648324], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "log2_ratio": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2386771887821483, 0.0, -0.16686818794607042, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.08746284125033942, 0.0, -0.4563782946597432, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6440501599927262, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9135542624944075, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.3924697494158373, 0.0, -0.08935259308703218, 0.0, -0.36200559427284795, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5195196782284426, 0.0, -0.38665249307649807, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.39474438269881157, -0.39231742277876036, -0.33562942750463615, -0.34792330342030686, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20090730426553627, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04999551707372642, -0.19161613901918262, -0.608610134909639], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}""",
    )
    generation_time = datetime.datetime(2020, 1, 1, 0, 0, 0)

    metric_definitions_data = io.StringIO(json.dumps({"test": "test"}))

    info = SampleInfo(
        pixelator_version="1.0.0",
        generation_date=generation_time.isoformat(),
        sample_id=sample_name,
        sample_description="just a test",
        panel_name="human-sc-immunology-spatial-proteomics",
        panel_version="1.0.0",
        technology="PNA",
        technology_version="2.5",
        parameters=[],
    )

    with open(report_path, "wb") as f:
        builder.write(
            f,
            sample_info=info,
            data=qc_report_data,
            metrics_definitions=metric_definitions_data,
        )

    assert report_path.exists()

    parser = lxml.etree.HTMLParser(huge_tree=True)
    with open(report_path, "rb") as f:
        document = lxml.html.parse(f, parser)
        body = document.find("body")

    metrics_data = builder.extract_field(body, "metrics")
    assert json.loads(metrics_data) == {
        "info": dataclasses.asdict(info),
        "metrics": qc_report_data.metrics,
    }

    ranked_component_size_data = builder.extract_field(body, "ranked-component-size")
    assert ranked_component_size_data == qc_report_data.ranked_component_size

    component_data = builder.extract_field(body, "component-data")
    assert component_data == qc_report_data.component_data

    antibody_percentages = builder.extract_field(body, "antibody-percentages")
    assert antibody_percentages == qc_report_data.antibody_percentages

    antibody_counts = builder.extract_field(body, "antibody-counts")
    assert antibody_counts == qc_report_data.antibody_counts

    proximity_data = builder.extract_field(body, "proximity-data")
    assert proximity_data == qc_report_data.proximity_data
