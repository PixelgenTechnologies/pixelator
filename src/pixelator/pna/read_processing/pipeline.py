"""Pipeline for processing amplicon reads, code adopted from cutadapt.

Modifications Copyright © 2025 Pixelgen Technologies AB.
Under the same license terms as the original code.

Original copyright notice:

Copyright (c) 2010 Marcel Martin <marcel.martin@scilifelab.se> and contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import typing
from typing import Iterable, Optional, Tuple

from cutadapt.files import InputFiles
from cutadapt.info import ModificationInfo
from cutadapt.modifiers import (
    PairedEndModifier,
    PairedEndModifierWrapper,
    SingleEndModifier,
)
from cutadapt.pipeline import Pipeline
from cutadapt.steps import PairedEndStep, SingleEndStep
from cutadapt.utils import Progress

from pixelator.pna.read_processing.modifiers import CombiningModifier

PairedModifiers = (
    PairedEndModifier | tuple[SingleEndModifier | None, SingleEndModifier | None]
)


ModifierStage = typing.Literal["pre", "post"]


class AmpliconPipeline(Pipeline):
    """Processing pipeline for processing single-end or paired‐end reads into a single amplicon sequence.

    - If two files (R1+R2) are provided: pre‐steps (paired), then combine, then post‐steps (single).
    - If only one file is provided (single‐end), you detect “R1” vs “R2” by looking at
      read.name, and treat that one FASTQ as if it were already “combined.”  All
      single‐end modifiers/steps run in that branch.

    :param combiner: The step that combines the reads into a single amplicon sequence.
    :param pre_modifiers: A list of modifiers that are applied to the reads before the combining step.
    :param pre_steps: A list of steps that are applied to the reads before the combining step.
    :param post_modifiers: A list of modifiers that are applied to the reads after the combining step.
    :param post_steps: A list of steps that are applied to the reads after the combining step.
    """

    paired: bool | None = None

    def __init__(
        self,
        combiner: CombiningModifier,
        pre_modifiers: (
            Iterable[
                PairedEndModifier
                | SingleEndModifier
                | tuple[
                    SingleEndModifier | None,
                    SingleEndModifier | None,
                ]
            ]
            | None
        ) = None,
        pre_steps: Iterable[PairedEndStep | SingleEndStep] | None = None,
        post_modifiers: Iterable[SingleEndModifier] | None = None,
        post_steps: Iterable[SingleEndStep] | None = None,
    ):
        """Initialize the pipeline.

        :param combiner: The step that combines the reads into a single
        amplicon sequence. In single-end mode, this step works as a modifier
        to enforce the expected read format.
        :param pre_modifiers: A list of modifiers that are applied to the reads before the combining step.
        :param pre_steps: A list of steps that are applied to the reads before the combining step.
        :param post_modifiers: A list of modifiers that are applied to the reads after the combining step.
        :param post_steps: A list of steps that are applied to the reads after the combining step.
        """
        self._combiner = combiner
        self._pre_modifiers: list[
            SingleEndModifier
            | PairedEndModifier
            | tuple[SingleEndModifier | None, SingleEndModifier | None]
        ] = []
        self._pre_steps: list[PairedEndStep | SingleEndStep] = (
            list(pre_steps) if pre_steps else []
        )
        self._post_modifiers: list[SingleEndModifier] = []
        self._post_steps: list[SingleEndStep] = list(post_steps) if post_steps else []

        self._reader = None

        # Whether to ignore pair_filter mode for discard-untrimmed filter
        self.override_untrimmed_pair_filter = False

        # Load the pre- and post-modifiers from the constructor arguments
        if pre_modifiers:
            self._add_modifiers(pre_modifiers, stage="pre")
        if post_modifiers:
            self._add_modifiers(post_modifiers, stage="post")

    @property
    def _modifiers(self) -> list[PairedEndModifier]:
        """Return a list of all modifiers in the pipeline."""
        return self._pre_modifiers + self._post_modifiers

    @property
    def _steps(self) -> list[PairedEndModifier]:
        """Return a list of all steps in the pipeline."""
        return self._pre_steps + [self._combiner] + self._post_steps

    def _add_modifiers(
        self,
        modifiers: Iterable[
            SingleEndModifier
            | PairedEndModifier
            | Tuple[Optional[SingleEndModifier], Optional[SingleEndModifier]],
        ],
        stage: ModifierStage,
    ) -> None:
        for modifier in modifiers:
            if isinstance(modifier, tuple):
                self._add_two_single_modifiers(modifier[0], modifier[1], stage=stage)
            else:
                self._add_modifier(modifier, stage=stage)

    def _add_two_single_modifiers(
        self,
        modifier1: Optional[SingleEndModifier],
        modifier2: Optional[SingleEndModifier],
        stage: ModifierStage,
    ) -> None:
        """Add two single-end modifiers that modify R1 and R2, respectively.

        One of them can be None, in which case the modifier
        is only applied to the respective other read.

        :param modifier1: The modifier for read 1.
        :param modifier2: The modifier for read 2.
        :param stage: The stage in which to apply the modifiers
        """
        assert stage in ("pre", "post")

        if modifier1 is None and modifier2 is None:
            raise ValueError("Not both modifiers can be None")

        if stage == "pre":
            self._pre_modifiers.append(PairedEndModifierWrapper(modifier1, modifier2))
        if stage == "post":
            self._post_modifiers.append(PairedEndModifierWrapper(modifier1, modifier2))

    def _add_modifier(self, modifier: PairedEndModifier, stage: ModifierStage) -> None:
        """Add a Modifier (without wrapping it in a PairedEndModifierWrapper)."""
        assert stage in ("pre", "post")

        if stage == "pre":
            self._pre_modifiers.append(modifier)
        if stage == "post":
            self._post_modifiers.append(modifier)

    def _pre_process_paired(self, reads):
        """Pre‐process paired reads."""
        pre_modifiers_and_steps = self._pre_modifiers + self._pre_steps
        read1, read2 = reads
        n_bp1 = len(read1)
        n_bp2 = len(read2)
        info1 = ModificationInfo(read1)
        info2 = ModificationInfo(read2)

        for step in pre_modifiers_and_steps:
            reads = step(*reads, info1, info2)  # type: ignore
            if reads is None:
                break

        if reads is not None:
            read1, read2 = reads

        return read1, read2, info1, info2, n_bp1, n_bp2

    def _pre_process_single(self, single_read):
        """Pre‐process single reads."""
        pre_modifiers_and_steps = self._pre_modifiers + self._pre_steps
        info = ModificationInfo(single_read)

        for modifier in pre_modifiers_and_steps:
            out = modifier(single_read, info)
            if out is None:
                break

        return out, info, len(single_read)

    def process_reads(
        self,
        infiles: InputFiles,
        progress: Optional[Progress] = None,
    ) -> Tuple[int, int, Optional[int]]:
        """Receive a slice of reads and process them through the pipeline.

        :param infiles: A list of input files to process.
        :returns: (n_reads, total1_bp, total2_bp or None).
        """
        self._infiles = infiles
        self._reader = infiles.open()

        if len(infiles._files) == 1:
            self.paired = False
        elif len(infiles._files) == 2:
            self.paired = True
        else:
            raise ValueError(
                "AmpliconPipeline requires either one or two input files (single‐end or paired‐end)."
            )

        n = 0  # no. of processed reads
        total1_bp = 0
        total2_bp = 0 if self.paired else None
        assert self._reader is not None

        post_modifiers_and_steps = self._post_modifiers + self._post_steps

        # Note that this reader is not a real file but a chunk backed by a BytesIO object
        # in case the pipeline is running on the ParallelPipelineRunner.
        # Each worker will have its own chunk of reads to process in its own pipeline instance.
        for reads in self._reader:
            n += 1
            if (n % 10000 == 0) and (progress is not None):
                progress.update(10000)
            if self.paired:
                read1, read2, info1, info2, n_bp1, n_bp2 = self._pre_process_paired(
                    reads
                )
                total1_bp += n_bp1
                total2_bp += n_bp2
                final_read = self._combiner(read1, read2, info1, info2)

            else:
                read, info, n_bp = self._pre_process_single(reads)
                total1_bp += n_bp
                # Figure out which read to pass as None, since the combiner expects two reads.
                if read.name.endswith("/2") or read.name.endswith("_R2"):
                    final_read = self._combiner(None, read, None, info)
                else:
                    final_read = self._combiner(read, None, info, None)

            final_read_info = ModificationInfo(final_read)

            if final_read is not None:
                for step in post_modifiers_and_steps:
                    try:
                        read = step(final_read, final_read_info)  # type: ignore
                    except Exception as e:
                        logging.error(
                            "Error in step %s for read %s", step, final_read.name
                        )
                        raise

                    if read is None:
                        break

        if progress is not None:
            progress.update(n % 10000)

        infiles.close()
        return (n, total1_bp, total2_bp)
