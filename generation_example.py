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