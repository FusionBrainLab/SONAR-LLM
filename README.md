# SONAR-LLM

This repository contains scripts for training Large Language Models (LLM) and SONAR-LLM on the **Tiny Stories** dataset.

SONAR-LLM is a sentence-level language model that uses the SONAR encoder and decoder to generate text from sentence embeddings; see [arXiv:2508.05305](https://arxiv.org/abs/2508.05305) for details.

Pretrained weights are available on Hugging Face in the SONAR-LLM collection: [raxtemur/collections/sonar-llm](https://huggingface.co/collections/raxtemur/sonar-llm-68d55c63cf38fc79bfa28179).

## 🚀 Quickstart: Generation

```python
from huggingface_hub import snapshot_download
from sonarllm_model import SONARLLMGenerator, SONARLLMGenerationConfig

p = snapshot_download("raxtemur/sonar-llm-300m")
gen = SONARLLMGenerator.load_from_checkpoint(p)
eos_emb = gen.t2vec.predict(["End of sequence."], source_lang="eng_Latn").to(gen.device)
cfg = SONARLLMGenerationConfig(
    temperature=0.2,
    latent_top_p=0.9,
    sentence_beam_size=2,
    latent_samples_per_step=4,
    max_sentences=4,
)

text, _ = gen.generate("Once upon a time.", eos_emb, cfg)
print(text)
```

---

## 📁 Available Scripts

### `train_llm.py`  
**Description**:  
Trains a Large Language Model (LLM) on the Tiny Stories dataset.

**Example command**:
```bash
torchrun --nproc_per_node=8 train_llm.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 8 \
  --output_dir OutName \
  --batch_size 8
```

---

### `train_sonarllm.py`  
**Description**:  
Legacy training script for SONAR-LLM. Encodes sentences **on-the-fly** using the Sonar encoder during training.  

**Example command**:
```bash
torchrun --nproc_per_node=8 train_sonarllm.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 64 \
  --output_dir OutName \
  --batch_size 1
```

---

### `train_sonarllm_fast.py`  
**Description**:  
Main training script for SONAR-LLM. Uses **precomputed** sentence embeddings generated with the Sonar encoder.

**Example command**:
```bash
torchrun --nproc_per_node=8 train_sonarllm_fast.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 64 \
  --output_dir OutName \
  --batch_size 1
```

---

### `sonar_encoding.py`  
**Description**:  
Generates sentence embeddings using the Sonar encoder. Required for `train_fast.py`.

**Example command**:
```bash
torchrun --nproc_per_node=8 sonar_encoding.py \
  --batch_size 32 \
  --output_dir ./sonar_embeddings_ts_100_shuffle
```

---

### `generate.py`  
**Description**:  
Generates full texts from Tiny Stories prefixes using a trained SONAR-LLM model.

---

### `evaluate_nlg.py`  
**Description**:  
Evaluates BLEU, ROUGE and METEOR metrics.

**Example command**:
```bash
python evaluate_nlg.py generated_900M.json
```

---

## ✅ Pretrained Models

See the Hugging Face collection for all released checkpoints: [SONAR-LLM collection](https://huggingface.co/collections/raxtemur/sonar-llm-68d55c63cf38fc79bfa28179).
