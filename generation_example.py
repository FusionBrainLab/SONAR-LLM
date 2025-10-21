from huggingface_hub import snapshot_download
from sonarllm_model import SONARLLMGenerator, SONARLLMGenerationConfig

p = snapshot_download("raxtemur/sonar-llm-1.3b")
gen = SONARLLMGenerator.load_from_checkpoint(p)
rus = "rus_Cyrl"
eng = "eng_Latn"

eos_emb_eng = gen.t2vec.predict(["End of sequence."], source_lang=eng).to(gen.device)
eos_emb_rus = gen.t2vec.predict(["End of sequence."], source_lang=rus).to(gen.device)

cfg = SONARLLMGenerationConfig(
    temperature=0,
    latent_top_p=0.9,
    sentence_beam_size=1,
    latent_samples_per_step=4,
    max_sentences=4
)

eng_text = "Petya loves Masha. Masha loves Gosha. Gosha loves Petya. Text summarization in one sentence only."
rus_text = "Петя любит Машу. Маша любит Гошу. Гоша любит Петю. Суммарное изложение в одном предложении."
print(eng_text)

cfg.target_lang = eng
cfg.source_lang = eng
text, _ = gen.generate(eng_text, eos_emb_eng, cfg)
print("English text answer (eng -> eng): ", text[len(eng_text)+1:])

cfg.target_lang = eng
cfg.source_lang = rus
text, _ = gen.generate(rus_text, eos_emb_eng, cfg)
print("English text answer (rus -> eng): ", text[len(rus_text)+1:])

cfg.target_lang = rus
cfg.source_lang = rus
text, _ = gen.generate(rus_text, eos_emb_rus, cfg)
print("Russian text answer (rus -> rus): ", text[len(rus_text)+1:])

cfg.target_lang = rus
cfg.source_lang = eng
text, _ = gen.generate(eng_text, eos_emb_rus, cfg)
print("Russian text answer (eng -> rus): ", text[len(eng_text)+1:])