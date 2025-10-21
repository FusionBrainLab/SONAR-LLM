from huggingface_hub import snapshot_download
from sonarllm_model import SONARLLMGenerator, SONARLLMGenerationConfig

p = snapshot_download("raxtemur/sonar-llm-1.3b")
gen = SONARLLMGenerator.load_from_checkpoint(p)
eos_emb = gen.t2vec.predict(["End of sequence."], source_lang="eng_Latn").to(gen.device)
cfg = SONARLLMGenerationConfig(
    temperature=0,
    latent_top_p=0.9,
    sentence_beam_size=1,
    latent_samples_per_step=4,
    max_sentences=4,
    target_lang="jpn_Jpan",
)

text, _ = gen.generate("Petya loves Masha. Masha loves Gosha. Gosha loves Petya. Text summarization in one sentence only.", eos_emb, cfg)
print(text)