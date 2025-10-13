import warnings
from typing import Iterable, List, Optional, Tuple, Union

import torch

from fairseq2.generation import Seq2SeqGenerator
from fairseq2.generation.beam_search.generator import BeamSearchSeq2SeqGenerator
from fairseq2.generation.sampling import Sampler, SamplingSeq2SeqGenerator

from sonar.inference_pipelines.utils import add_progress_bar, SequenceToTextConverter
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline as _BaseEmbeddingToTextModelPipeline,
)
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.nn import BatchLayout

class EmbeddingToTextModelPipeline(_BaseEmbeddingToTextModelPipeline):
    """Drop-in replacement that can also return sentence log-probabilities via return_scores.

    - When return_scores=False (default), behaves exactly like the base pipeline and returns List[str].
    - When return_scores=True, returns a tuple (List[str], List[float]) where each float is the
      hypothesis score from fairseq2 (sum of token log-probabilities if normalize_scores=False,
      otherwise length-normalized per fairseq2 semantics).
    """

    @torch.inference_mode()
    def predict(
        self,
        inputs: torch.Tensor,
        target_lang: str,
        target_mode: str = "target",
        batch_size: int = 5,
        progress_bar: bool = False,
        sampler: Optional[Sampler] = None,
        return_scores: bool = False,
        **generator_kwargs,
    ) -> List[str]:
        if sampler is not None:
            generator: Seq2SeqGenerator = SamplingSeq2SeqGenerator(
                self.model, self.tokenizer.vocab_info, sampler, **generator_kwargs
            )
        else:
            generator = BeamSearchSeq2SeqGenerator(
                self.model, self.tokenizer.vocab_info, **generator_kwargs
            )

        converter = SequenceToTextConverter(
            generator,
            self.tokenizer,
            task="translation",
            target_lang=target_lang,
            mode=target_mode,
        )

        def _do_translate(src_tensors: List[torch.Tensor]) -> Union[List[str], Tuple[List[str], List[float]]]:
            seqs = torch.stack(src_tensors).to(self.device)
            seqs_layout = BatchLayout.of(seqs)
            texts, gen_out = converter.batch_convert(seqs, seqs_layout)
            if return_scores:
                scores: List[float] = []
                for hyps in gen_out.hypotheses:
                    if len(hyps) == 0 or hyps[0].score is None:
                        scores.append(0.0)
                    else:
                        scores.append(float(hyps[0].score))
                return texts, scores
            return texts

        pipeline: Iterable = (
            read_sequence(list(inputs))
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=inputs, batch_size=batch_size)

        results: List = list(iter(pipeline))

        if not return_scores:
            # results is List[List[str]] → flatten
            return [text for batch_texts in results for text in batch_texts]

        # results is List[Tuple[List[str], List[float]]] → flatten both
        all_texts: List[str] = []
        all_scores: List[float] = []
        for batch in results:
            batch_texts, batch_scores = batch
            all_texts.extend(batch_texts)
            all_scores.extend(batch_scores)
        return all_texts, all_scores


