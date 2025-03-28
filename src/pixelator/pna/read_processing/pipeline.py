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
    """Processing pipeline for combining paired-end reads into a single amplicon sequence.

    A series of modifiers and steps can be applied to the reads before and after the combining step.
    The pipeline is designed to work with paired-end reads only.

    :param combiner: The step that combines the reads into a single amplicon sequence.
    :param pre_modifiers: A list of modifiers that are applied to the reads before the combining step.
    :param pre_steps: A list of steps that are applied to the reads before the combining step.
    :param post_modifiers: A list of modifiers that are applied to the reads after the combining step.
    :param post_steps: A list of steps that are applied to the reads after the combining step.
    """

    paired = True

    def __init__(
        self,
        combiner: CombiningModifier,
        pre_modifiers: Iterable[PairedEndModifier] | None = None,
        pre_steps: Iterable[PairedEndStep] | None = None,
        post_modifiers: Iterable[SingleEndModifier] | None = None,
        post_steps: Iterable[SingleEndStep] | None = None,
    ):
        """Create a new AmpliconPipeline instance.

        :param combiner: The step that combines the reads into a single amplicon sequence.
        :param pre_modifiers: A list of modifiers that are applied to the reads before the combining step.
        :param pre_steps: A list of steps that are applied to the reads before the combining step.
        :param post_modifiers: A list of modifiers that are applied to the reads after the combining step.
        :param post_steps: A list of steps that are applied to the reads after the combining step.
        """
        self._combiner = combiner
        self._pre_modifiers: list[PairedEndModifier] = []
        self._pre_steps: list[PairedEndStep] = list(pre_steps) if pre_steps else []
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
        return self._pre_steps or [] + [self._combiner] + self._post_steps or []

    def _add_modifiers(self, modifiers, stage: ModifierStage):
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

    def process_reads(
        self,
        infiles: InputFiles,
        progress: Optional[Progress] = None,
    ) -> Tuple[int, int, Optional[int]]:
        """Receive a slice of reads and process them through the pipeline.

        :param infiles: A list of input files to process.
        """
        self._infiles = infiles
        self._reader = infiles.open()
        n = 0  # no. of processed reads
        total1_bp = 0
        total2_bp = 0
        assert self._reader is not None

        pre_modifiers_and_steps = self._pre_modifiers + self._pre_steps
        post_modifiers_and_steps = self._post_modifiers + self._post_steps

        # Note that this reader is not a real file but a chunk backed by a BytesIO object
        # in case the pipeline is running on the ParallelPipelineRunner.
        # Each worker will have its own chunk of reads to process in its own pipeline instance.
        for reads in self._reader:
            n += 1
            if n % 10000 == 0 and progress is not None:
                progress.update(10000)
            read1, read2 = reads
            total1_bp += len(read1)
            total2_bp += len(read2)
            info1 = ModificationInfo(read1)
            info2 = ModificationInfo(read2)

            for step in pre_modifiers_and_steps:
                reads = step(*reads, info1, info2)  # type: ignore
                if reads is None:
                    break

            if reads is not None:
                read1, read2 = reads
                combined_read = self._combiner(read1, read2, info1, info2)
                combined_read_info = ModificationInfo(combined_read)

                if combined_read is not None:
                    for step in post_modifiers_and_steps:
                        try:
                            read = step(combined_read, combined_read_info)  # type: ignore
                        except Exception as e:
                            logging.error(
                                "Error in step %s for read %s", step, combined_read.name
                            )
                            raise

                        if read is None:
                            break

        if progress is not None:
            progress.update(n % 10000)

        infiles.close()
        return (n, total1_bp, total2_bp)
