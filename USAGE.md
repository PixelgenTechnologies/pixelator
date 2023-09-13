# Usage

Pixelator is composed of different commands (stages), which are prefixed by the assay
type that you want to run it for. For example, if you want to run the `graph` command
for a single-cell assay you can do so by running:

````shell
pixelator single-cell graph ...
````

The stages are  designed to be run sequentially but can also be run separately (for
instance if one wants to re-run a specific stage/command or resume a failed job).
The main commands are: `amplicon`, `preqc`, `adapterqc`, `demux`, `collapse`,
`graph` and `annotate`. These commands are the minimum steps required to process
the raw sequencing data and obtain the processed data structures, metrics and figures.

The `analysis` command can be used to perform downstream analyses and it uses
the output of the `annotate` command.

The `report` command can be used to generate HTML web reports of a dataset
(one or several samples). This command can be run if all the main commands have
been previously performed successfully.

An example of how to run the main commands sequentially is provided here:

````shell
pixelator single-cell amplicon --output test /path/to/rawdata/*.fastq.gz

pixelator single-cell preqc --design D12 --output test /test/amplicon/*.fastq.gz

pixelator single-cell adapterqc --design D12 --output test test/preqc/*processed.fastq.gz

pixelator single-cell demux --panel human-sc-immunology-spatial-proteomics --design D12 --output test test/adapterqc/*processed.fastq.gz

pixelator single-cell collapse --design D12 --output test test/demux/*processed*.fastq.gz

pixelator single-cell graph --output test test/collapse/*collapsed.csv.gz

pixelator single-cell annotate --panel human-sc-immunology-spatial-proteomics --output test test/graph/*.edgelist.csv.gz

pixelator single-cell analysis --output test test/annotate/*.pxl

pixelator single-cell report --panel human-sc-immunology-spatial-proteomics --output test test
````

Each command has specific parameters. You can type `pixelator command_name --help` to get
a description of the command's arguments. The main command `pixelator` has some arguments that
are passed to all the commands (`--log-file`, `--verbose` and `--cores`).

There are preloaded design (kit) settings with the most important settings for different assays.
These are required in the `preqc`, `adapterqc`, `demux` and `collapse` commands (--design).

The list of designs available can be seen by typing `pixelator single-cell --list-designs`.
The settings can be overruled by using the respective arguments in the different commands but
you should do this with care.

Each command will generate a subfolder inside the provided output folder (`--output`).
We **strongly** encourage to use the same output folder for all the stages (commands).
This also allows to re-run any step at any moment. The sample name (id) as in the raw
input files names is kept in the generated files in all the commands.

The `amplicon` command must be used if you are working either with paired-end (PE) data
or single-end (SE) data. It will combine any MPX supported read design into a single merged
amplicon file (before the `preqc` command):

````shell
pixelator single-cell amplicon --output test /path/to/rawdata/*.fastq.gz
````

If your sequencing is a pair-end MPX design, the fastq files must contain identifiers for
pixelator to be able to distinguish R1(FW) (e.g. default: `_R1`) from R2(RV) (e.g.
default: `_R2`) and these can be passed to `amplicon` trough the arguments `--input1-pattern`
and `--input2-pattern`. The input files must be of the same size (number of reads) and their
reads must be in the same order.

The output files will be placed in a folder called `amplicon` inside the output folder.

The `preqc` command performs QC and quality filtering of the raw sequencing data (FASTQ).
It also generates a QC report in HTML and JSON formats. It saves processed reads
as well as discarded reads (*"too short"* or *"too many Ns"* or *"too low quality"*, etc.).

If you use the flag `--dedup` the duplicated reads will be removed. This will make
the whole pipeline faster and less memory intensive but the real number of molecules
("count" column) in the edge list will be lost.

The output files will be placed in a folder called `preqc` inside the output folder.

The `adapterqc` command performs a sanity check on the correctness/presence of the PBS1/2 sequences.
It also generates a QC report in JSON format. It saves processed reads as well as discarded reads
(with no match to PBS1/2).

The output files will be placed in a folder called `adapterqc` inside the output folder.

The `demux` command assigns a marker (barcode) to each read. It also generates QC
report in JSON format. It saves processed reads (one per antibody) as well as discarded reads
(with no match to given barcodes/antibodies). In this step an antibody panel file (CSV) or key to
Pixelgen Technologies panels is required (`--panel`). This file contains the antibodies present
in the data as well as their sequences and it needs the following columns:

    marker_id,control,nuclear,full_name,alt_id,sequence,conj_id

You can find a list of antibody panels [here](https://github.com/PixelgenTechnologies/pixelgen-panels)
(please make sure to use the correct panel for your data).

You can use (`--rev-complement`) if you want to use the reverse complement sequence of the antibody
and (`--anchored`) if you want to anchor sequences to the right most position in the read
(See [cutadapt's documentation](https://cutadapt.readthedocs.io/en/stable/guide.html) for more details).
However, these two settings are predefined with the design (`--design`) and should only be changed with care.

The output files will be placed in a folder called `demux` inside the output folder.

The `collapse` command removes duplicates and performs error correction. This is
achieved using the UPI and UMI sequences to check for uniqueness, collapse and compute
a read count. The command generates a QC report in JSON format. Errors are allowed
when collapsing reads using different collapsing algorithms (`--algorithm`). The output
format of this command is an edge list dataframe in CSV format:

    upia,upib,umi,marker,sequence,count,umi_unique_count,upi_unique_count

Note that the `collapse` command may have high memory requirements when processing large
datasets (specially for antibodies with many reads as processing is done in parallel
per antibody). There are different options that can be used alone or combined
in order to decrease the memory usage:

- Use `--algorithm unique` which will disable the error correction
- Use `--min-count` with a value of 2 for example to remove singletons
- Use `--dedup` in pixelator `preqc`, which will remove duplicates

The output files will be placed in a folder called `collapse` inside the output folder.

The `graph` command takes as input the edge list dataframe (CSV) generated in the collapse step and
after filtering it by count (`--min-count`) the connected components of the graph (graphs) are computed
and added to the edge list in a column called "component".

The `graph` command has the option to recover components (technical multiplets) into smaller components
using community detection to detect and remove problematic edges. (See `--multiplet-recovery`).
The information to keep track of the original and new (recovered) components is stored in a file
(components_recovered.csv). An edge list containing only the removed edges is written to a CSV file
(discarded_edgelist.csv.gz).

The following files are generated in the graph command:

    - raw edge list dataframe (CSV) before recovering technical multiplets
    - edge list dataframe (CSV) after recovering technical multiplets
    - metrics (JSON) with useful information about the clustering

Note that if the `--multiplet-recovery` is not active the raw and recovered edge list
will be the same. If you use the `--multiplet-recovery` option the memory requirements may increase
and thus it is recommended to use less cores (`--cores`) in case you are processing multiple samples
in a single machine.

The output files will be placed in a folder called `graph` inside the output folder.

The `annotate` command takes as input the edge list (CSV) file generated in the graph command.
The command then performs filtering and cell calling of the components. Optionally, if `--cell-annotation`
is active, the edge list is converted to an `AnnData` object, and annotated into major PBMC cell types.

In this step an antibody panel file (CSV) is required (`--panel`) as described in the `demux`
command.

The AnnData file will have the same dimension as in the antibody panel so any missing antibody
will be filled with 0's.

The output AnnData will contain the following structure:

    .X = the component to antibody counts
    .var = the antibody metrics
    .obs = the component metrics
    .obsm["normalized_rel"] = the normalized (REL by component) component to antibody counts
    .obsm["clr"] = the transformed (CLR by component) component to antibody counts
    .obsm["log1p"] = the transformed (log1p) component to antibody counts
    .obsm["denoised"] = the denoised (CLR by component) counts if control antibodies are present

The annotate command allows you to either set manual limits for component sizes
with `--min-size` and `--max-size`, or to enable a dynamic size filter (min, max or both)
with `--dynamic-filter`. This implements a rank-based method to try to find
the distribution of putative cells (lower/upper bound).

The annotate command will perform dimensionality reduction and unsupervised clustering
using the CLR-transformed antibody counts. These will be added to the `leiden` and
`X_umap` variables in `obs` and `obsm` respectively.

The annotate command will call aggregates (when enabled using `--aggregate-calling`).
This will add two keys to the `obs` part of the AnnData:

    - tau_type: components will be marked as "normal", "high" or "low". The "normal"
      category indicates that the component is not an aggregate, while the "high" and
      "low" categories indicate that the component is likely an aggregate and should be
      filtered from downstream analysis in most cases
    - tau: aggregation specificity score computed for the component

In addition to this the limits used to call components as having a "high" or "low" tau type
([1] , [2]) will be added to `uns["tau_thresholds"]`.

The output files will be placed in a folder called `annotate` inside the output folder.

The following files are generated:

    - A dataframe with the components metrics before filtering (CSV)
    - PixelDataset (PXL) with the filtered AnnData and edge list
    - metrics (JSON) with useful information about the annotation

The PixelDataset is a zip bundle with the AnnData (`adata.h5ad`) and the edge list
(`edgelist.csv.gz`) files.

The `analysis` command can be used to perform downstream analysis and requires the
annotate command to have been completed. The input of the analysis command is a
`PixelDataset` in PXL format generated in the annotate command.

Currently the following analysis can be performed (if enabled):

    - polarization scores (all the statistics in a dataframe)
    - co-localization scores (all pair-wise scores in a dataframe)

The polarization scores is a dataframe with the following columns:

    - morans_i
    - morans_p_value
    - morans_p_value_adjusted
    - morans_z
    - marker
    - component

The polarization scores are computed using
[Moran's spatial autocorrelation](https://en.wikipedia.org/wiki/Moran%27s_I).
A high score should indicate that the antibody has a localized spatial
pattern.

The colocalization scores is a dataframe with the following columns:

    - marker_1
    - marker_2
    - pearson
    - pearson_z
    - pearson_p_value
    - pearson_p_value_adjusted
    - jaccard
    - jaccard_z
    - jaccard_p_value
    - jaccard_p_value_adjusted
    - component


The `jaccard` scores are computed using a
[Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index>) on the binary counts
and they should indicate that the two antibodies are located in the same area.

The `pearson` scores are computed using the
[Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
on the counts and they should indicate that the two antibodies are located in the same
area and with the similar abundance.

The output of the `analysis` command is a `PixelDataset` updated with the
respective scores (the ones that are enabled when running the command).

The analysis command allows to apply different normalization options (raw counts,
CLR-transformation and denoise). See `pixelator single-cell analysis --help` for
more information.

A common usage example for this command is:

````shell
pixelator single-cell analysis --compute-polarization --output test test/annotate/*.dataset.pxl
````

The output files will be placed in a folder called `analysis` inside the output folder.

The `report` command takes as input a folder where all the main steps
have been performed (`amplicon`, `preqc`, `adapterqc`, `demux`, `collapse`, `graph` and `annotate`)
and generates a web report (HTML) with summary stats, metrics and interactive plots for
each sample (HTML). An example on how to generate a report:

````shell
pixelator single-cell report --panel human-sc-immunology-spatial-proteomics --output test test
````

The output files will be placed in a folder called `report` inside the output folder.

The report command can take an optional metadata file in CSV format (--metadata). This file must
contain the following fields (comma separated):

    sample_id,sample_description,panel_version,panel_name

The information in the metadata file will be included in the web reports.
The sample_id field must match the sample names in the data.

## Pixelator Docker Images

Pixelator is automatically packaged in a Docker container available from the several container registries:

```shell
docker pull ghcr.io/pixelgentechnologies/pixelator:latest
docker run pixelator --help
```

Make sure that all input and output paths are accessible to the container and host system by mounting
the directories containing these paths. All input/output paths must be passed as absolute paths with -v.

For example:

```shell
docker run -v /home/myuser:/home/myuser pixelator single-cell amplicon --output /home/myuser/run /home/myuser/data/Sample*fastq.gz
```

### References

[1]. Yanai, I. et al. Genome-wide midrange transcription profiles reveal expression level relationships in human tissue specification. Bioinformatics, Volume 21, Issue 5, March 2005, Pages 650–659, https://doi.org/10.1093/bioinformatics/bti042

[2]. Kryuchkova-Mostacci, N. and Robinson-Rechavi, M. A benchmark of gene expression tissue-specificity metrics. Briefings in Bioinformatics, Volume 18, Issue 2, March 2017, Pages 205–214, https://doi.org/10.1093/bib/bbw008

[1]: https://doi.org/10.1093/bioinformatics/bti042 "Yanai, I. et al. Genome-wide midrange transcription profiles reveal expression level relationships in human tissue specification. Bioinformatics, Volume 21, Issue 5, March 2005, Pages 650–659"

[2]: https://doi.org/10.1093/bib/bbw008 "Kryuchkova-Mostacci, N. and Robinson-Rechavi, M. A benchmark of gene expression tissue-specificity metrics. Briefings in Bioinformatics, Volume 18, Issue 2, March 2017, Pages 205–214"
