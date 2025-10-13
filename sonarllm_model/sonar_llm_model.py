import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union


import nltk
from nltk.tokenize import sent_tokenize
from transformers.cache_utils import Cache
nltk.download("punkt", quiet=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq2

# Prefer the new pipeline API when available; fall back to legacy otherwise
try:
    from .embedding_to_text_with_scores import EmbeddingToTextModelPipeline  # type: ignore
except Exception:
    from .embedding_to_text_with_scores_old import EmbeddingToTextModelPipeline  # type: ignore

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from .samplers.direction import build_direction_sampler
import json
import os
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.to(self.device))
    
    def to(self, device: torch.device) -> "Projector":
        self.device = device
        self.linear.to(device)
        return self

class BeamState(NamedTuple):
    sents: List[str]
    emb: torch.Tensor
    score: float = 0.0
    recent_dirs: List[torch.Tensor] = []
    past_key_values: Optional[Cache] = None
    inconsistencies: List[Tuple[float, torch.Tensor]] = []

@dataclass
class SONARLLMGenerationConfig:
    # Outer sentence-level beam
    sentence_beam_size: int = 1
    latent_samples_per_step: int = 4  # M latent variants per active beam state

    # Token-level decoder params
    decoder_beam_size: int = 5  # default in fairseq2
    decoder_temperature: float = 1.0  # default in fairseq2
    normalize_sentence_scores: bool = True  # False → sum of token log-probs
    decoder_max_len: int = 256

    # Latent sampling
    temperature: float = 0
    latent_top_p: Optional[float] = None  # 0<p<=1 or None for Gaussian
    temperature_mode: str = "relative"  # "absolute" | "relative"

    # Repetition control in latent space
    repetition_penalty: float = 0.0
    repetition_memory: int = 0

    # Direction sampling strategy
    # Options: "isotropic", "mean_shift_penalty", "vector_angle_penalty"
    direction_sampler: str = "isotropic"
    # whether to update the base vector or the noise vector (ignored by isotropic)
    to_base_vector: bool = True
    to_noise_vector: bool = False

    # Termination
    max_sentences: int = 32
    eos_threshold: float = 0.98

    # in development
    use_cache: bool = False


class SONARLLMGenerator(torch.nn.Module):
    """Sentence-level beam over latent reversed embeddings using SONAR decoder.

    For each step:
    - Run LLaMA on the sentence embedding history to get final hidden.
    - Sample multiple latent directions (temperature/latent_top_p, with repetition penalty).
    - Project to `reversed_emb` and decode text via SONAR decoder.
    - Score each candidate using decoder sentence logprob (+ optional shaping).
    - Keep top `sentence_beam_size` states and continue until EOS or max sentences.

    This class does NOT modify existing project files and can be used standalone.
    """

    def __init__(
        self,
        llama_model: nn.Module,
        forward_proj: nn.Module,
        reverse_proj: nn.Module,
        sonar_decoder: EmbeddingToTextModelPipeline,
        t2vec_model: TextToEmbeddingModelPipeline,
        device: torch.device,
        add_begin: bool = False,
    ) -> None:
        super().__init__()
        self.llama_model = llama_model.eval()
        self.forward_proj = forward_proj.eval()
        self.reverse_proj = reverse_proj.eval()
        self.sonar_decoder = sonar_decoder.eval()
        self.t2vec = t2vec_model.eval()
        self.device = device
        self.add_begin = add_begin

    @torch.no_grad()
    def generate(self, prefix_text: str, eos_emb: torch.Tensor, cfg: Optional[SONARLLMGenerationConfig] = None) -> str:
        eos_emb = eos_emb.to(self.device)
        # Normalize and attach config to the instance for helper use
        if cfg is None:
            cfg = SONARLLMGenerationConfig()

        if cfg.use_cache:
            print("use_cache is not supported yet")
            # BUG if hs was sampled -> we cannot use old past_key_values
            # probably we can use pre-last past_key_values -- they do not include last hs which was sampled
            cfg.use_cache = False

        self._cfg = cfg

        # Build direction sampler and optionally rotate base hidden (no norm change)
        self.penalty_sampler = build_direction_sampler(
            getattr(self._cfg, "direction_sampler", "mean_shift_penalty"),
            repetition_penalty=float(self._cfg.repetition_penalty) if self._cfg.repetition_penalty is not None else 0.0,
            repetition_memory=int(self._cfg.repetition_memory) if self._cfg.repetition_memory is not None else 0,
            cfg=self._cfg,
        )

        sents = sent_tokenize(prefix_text)
        if self.add_begin:
            sents = ["Begin of text."] + sents

        if len(sents) == 0:
            sents = [prefix_text.strip()]

        # Initialize prefix embeddings
        emb_seq = self.t2vec.predict(sents, source_lang="eng_Latn").to(self.device)

        # Beam state tuple: (sentences, embeddings_seq, cumulative_score, recent_dirs)
        beams: List[BeamState] = [
            BeamState(sents[:], emb_seq)
        ]

        steps = 0
        while steps < self._cfg.max_sentences:
            steps += 1
            candidates: List[BeamState] = []

            for beam in beams:
                candidates.extend(
                    self._expand_beam_state(beam)
                )

            # Keep top-k beams
            if len(candidates) == 0:
                break
            candidates.sort(key=lambda b: b[2], reverse=True)
            beams = candidates[: int(self._cfg.sentence_beam_size)]

            # If all beams look ended by EOS threshold, stop early
            if self._all_close_to_eos(beams, eos_emb):
                break

        best = max(beams, key=lambda b: b[2])
        result = self._join_sentences(best[0])
        if self.add_begin:
            result = result[len("Begin of text."):]
        return result, best

    # --- internals ---
    def _expand_beam_state(
        self,
        beam: BeamState,
    ) -> List[BeamState]:
        hist_sents, hist_emb, score, recent_dirs, past_key_values, inconsistencies = beam
        """Expand one beam state into candidate next states.

        Returns a list of BeamState including (new_hist_sents, new_hist_emb, new_score, new_recent_dirs, ...).
        """
        final_hidden, past_key_values = self._forward_hidden(hist_emb, past_key_values)
        base_hidden = self.penalty_sampler.update_base_vector(final_hidden, recent_dirs)
        out: List[BeamState] = []

        for _ in range(max(1, int(self._cfg.latent_samples_per_step))):
            dir_unit = self.penalty_sampler.sample_unit_direction(base_hidden, recent_dirs, rng=None)
            noise = self._sample_noise(base_hidden, dir_unit)
            h_perturbed = base_hidden + noise
            z = self.reverse_proj(h_perturbed.unsqueeze(0))

            texts, scores = self.sonar_decoder.predict(
                z,
                target_lang="eng_Latn",
                beam_size=int(self._cfg.decoder_beam_size),
                normalize_scores=bool(self._cfg.normalize_sentence_scores),
                max_seq_len=self._cfg.decoder_max_len,
                temperature=float(self._cfg.decoder_temperature),
                return_scores=True,
            )
            text = texts[0]

            z_re = self.t2vec.predict([text], source_lang="eng_Latn").to(self.device)
            # z_re = self.forward_proj(z_re.unsqueeze(0)) if z_re.ndim == 2 else self.forward_proj(z_re)
            difference = z_re - z
            cos = F.cosine_similarity(z_re, z, dim=1).item()
            inconsistency = (cos, difference)

            sent_logprob = float(scores[0])
            cand_score = score + sent_logprob
            new_recent = self.penalty_sampler.update_recent_dirs(recent_dirs, dir_unit)

            new_hist_sents = hist_sents + [text]
            new_hist_emb = torch.cat([hist_emb, z_re], dim=0)
            
            if self._cfg.use_cache:
                new_past_key_values = past_key_values
            else:
                new_past_key_values = None

            out.append(BeamState(new_hist_sents, new_hist_emb, cand_score, new_recent, new_past_key_values, inconsistencies + [inconsistency]))

        return out

    def _all_close_to_eos(self, beams, eos_emb: torch.Tensor) -> bool:
        for beam in beams:
            last = beam.emb[-1:, :]
            sim = F.cosine_similarity(last, eos_emb, dim=1).item()
            if sim < float(self._cfg.eos_threshold):
                return False
        return True

    @torch.no_grad()
    def _forward_hidden(self, emb_seq: torch.Tensor, past_key_values: Optional[Cache] = None) -> torch.Tensor:
        proj = self.forward_proj(emb_seq.unsqueeze(0)) if emb_seq.ndim == 2 else self.forward_proj(emb_seq)
        out = self.llama_model(inputs_embeds=proj, output_hidden_states=True, past_key_values=past_key_values)
        hidden = out.hidden_states[-1]
        return hidden[0, -1, :], out.past_key_values

    def _join_sentences(self, sents: List[str]) -> str:
        return " ".join(sents)

    def _update_recent_dirs(
        self, recent: List[torch.Tensor], u: torch.Tensor, memory_cap: int
    ) -> List[torch.Tensor]:
        if memory_cap <= 0:
            return recent
        if not torch.isfinite(u).all():
            return recent
        new_recent = recent + [u.detach().to("cpu")]
        if len(new_recent) > int(memory_cap):
            new_recent = new_recent[-int(memory_cap) :]
        return new_recent

    def _sample_noise(
        self, final_hidden: torch.Tensor, dir_unit: torch.Tensor
    ) -> torch.Tensor:
        t = float(self._cfg.temperature)
        if t <= 0.0:
            return torch.zeros_like(final_hidden)

        if self._cfg.temperature_mode not in ("absolute", "relative"):
            raise ValueError(f"Unsupported temperature_mode: {self._cfg.temperature_mode}")

        if self._cfg.temperature_mode == "absolute":
            sigma = torch.tensor(t, device=final_hidden.device, dtype=final_hidden.dtype)
        else:
            rms = torch.sqrt(torch.mean(final_hidden.to(torch.float32) ** 2))
            rms = torch.clamp(rms, min=1e-12).to(dtype=final_hidden.dtype, device=final_hidden.device)
            sigma = rms * t

        top_p = self._cfg.latent_top_p
        if top_p is None:
            top_p = 1.0
        return self._sample_truncated_normal_like(final_hidden, float(top_p), sigma, dir_unit)

    def _sample_truncated_normal_like(
        self, base_vector: torch.Tensor, top_p: float, sigma: torch.Tensor, dir_unit: torch.Tensor
    ) -> torch.Tensor:
        # Wilson–Hilferty approximation for ChiSquare quantiles
        dim = base_vector.numel()
        device = base_vector.device
        u = torch.rand((), device=device, dtype=torch.float32)
        p = torch.clamp(u * float(top_p), min=1e-12, max=1.0 - 1e-12)
        k = torch.tensor(float(dim), device=device, dtype=torch.float32)
        z = torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float32)) * torch.special.erfinv(2.0 * p - 1.0)
        term = 1.0 - 2.0 / (9.0 * k) + z * torch.sqrt(2.0 / (9.0 * k))
        term = torch.clamp(term, min=1e-12)
        s = k * (term ** 3)
        r = torch.sqrt(torch.clamp(s, min=1e-12)).to(dtype=base_vector.dtype)
        return dir_unit * (r * sigma)


    # --- factory ---
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
        generation_config: Optional[SONARLLMGenerationConfig] = None,
    ) -> "SONARLLMGenerator":
        """Load generator from a folder with config.json and weights.

        The folder is expected to contain:
          - config.json (with keys: pretrained_model_name_or_path, llama_config?, embed_dim)
          - pytorch_model.bin (or model_state_dict inside the saved file)
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg_path = os.path.join(checkpoint_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model_name_or_path"])
        tokenizer.pad_token = tokenizer.eos_token

        llama_cfg_dict = cfg.get("llama_config", {})
        if "vocab_size" not in llama_cfg_dict:
            llama_cfg_dict["vocab_size"] = len(tokenizer)
        # llama_cfg_dict["pad_token_id"] = tokenizer.pad_token_id
        # llama_cfg_dict["bos_token_id"] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000
        # llama_cfg_dict["eos_token_id"] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001
        llama_cfg = LlamaConfig(**llama_cfg_dict) if "llama_config" in cfg else LlamaConfig()

        llama_model = LlamaForCausalLM(llama_cfg).to(device).eval()

        hidden_size = llama_cfg.hidden_size
        embed_dim = cfg.get("embed_dim", 1024)

        t2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        ).eval()

        vec2text_model = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        ).eval()

        forward_projector = Projector(embed_dim, hidden_size).to(device).eval()
        reverse_projector = Projector(hidden_size, embed_dim).to(device).eval()

        gen = cls(
            llama_model,
            forward_projector,
            reverse_projector,
            vec2text_model,
            t2vec_model,
            device,
            add_begin=cfg.get("add_begin", False),
        )

        # Load weights into generator to cover llama + projectors
        ckpt_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
        state = torch.load(ckpt_bin, map_location=device, weights_only=True)
        state = state.get("model_state_dict", state)
        raw = gen.module if hasattr(gen, "module") else gen
        missing, unexpected = raw.load_state_dict(state, strict=False)
        # try:
        #     n_miss = len(missing) if hasattr(missing, "__len__") else 0
        #     n_unexp = len(unexpected) if hasattr(unexpected, "__len__") else 0
        #     if n_miss or n_unexp:
        #         print(f"[new loader] missing={n_miss}, unexpected={n_unexp}")
        #         if n_miss:
        #             print("  missing (first 10):", list(missing)[:10])
        #         if n_unexp:
        #             print("  unexpected (first 10):", list(unexpected)[:10])
        #         if n_miss:
        #             miss_fwd = [k for k in missing if str(k).startswith('forward_proj.')]
        #             miss_rev = [k for k in missing if str(k).startswith('reverse_proj.')]
        #             miss_llm = [k for k in missing if str(k).startswith('llama_model.')]
        #             print(f"  missing counts: forward_proj={len(miss_fwd)}, reverse_proj={len(miss_rev)}, llama_model={len(miss_llm)}")
        # except Exception:
        #     pass

        return gen


    def to(self, device: Union[torch.device, str]) -> "SONARLLMGenerator":
        super().to(device)
        torch.cuda.set_device(device)
        if type(device) == str:
            device = torch.device(device)
        self.device = device
        self.llama_model.to(self.device)
        self.device = self.llama_model.device
        self.forward_proj.to(self.device)
        self.reverse_proj.to(self.device)
        self.sonar_decoder.device = self.device
        self.t2vec.to(self.device)
        self.t2vec.device = self.device
        return self