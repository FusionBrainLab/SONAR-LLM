import argparse
import json
import statistics
import sys
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure that the Punkt tokenizer model is available
nltk.download("punkt", quiet=True)

# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def evaluate_text(generated_text: str, reference_text: str):
    """
    Return BLEU and ROUGE-L_f scores for a pair of strings.
    """
    gen_tokens = generated_text.split()
    ref_tokens = reference_text.split()

    bleu = sentence_bleu(
        [ref_tokens],
        gen_tokens,
        smoothing_function=SmoothingFunction().method4,
    )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, generated_text)
    rouge_l_f = rouge_scores["rougeL"].fmeasure

    return bleu, rouge_l_f


def evaluate_meteor(generated_text: str, reference_text: str):
    """
    Return the METEOR score for a pair of strings.
    """
    return meteor_score([reference_text.split()], generated_text.split())


# --------------------------------------------------------------------------- #
# Process a single JSON record
# --------------------------------------------------------------------------- #
def process_record(rec: dict):
    prefix = rec["prefix"]
    gen_full = rec["full_text"]
    ref_full = rec["original_text"]

    # Assert that both texts start with the prefix, then strip it
    assert gen_full.startswith(prefix), "full_text does not start with its prefix"
    assert ref_full.startswith(prefix), "original_text does not start with its prefix"

    gen_full = gen_full[len(prefix) :]
    ref_full = ref_full[len(prefix) :]

    # First sentence only
    gen_sents = sent_tokenize(gen_full, language="english") if gen_full else []
    ref_sents = sent_tokenize(ref_full, language="english") if ref_full else []
    gen_sent = gen_sents[0] if gen_sents else ""
    ref_sent = ref_sents[0] if ref_sents else ""

    bleu, rouge_l = evaluate_text(gen_sent, ref_sent)
    meteor = evaluate_meteor(gen_sent, ref_sent)
    return bleu, rouge_l, meteor


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate texts stored in a JSON file.")
    parser.add_argument("input_json", help="Path to the JSON file to evaluate")
    args = parser.parse_args()

    path = Path(args.input_json)
    if not path.exists():
        sys.exit(f"File {path} not found.")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    bleu_list, rouge_list, meteor_list = [], [], []
    for rec in data:
        bleu, rouge_l, meteor = process_record(rec)
        bleu_list.append(bleu)
        rouge_list.append(rouge_l)
        meteor_list.append(meteor)

    if not bleu_list:
        sys.exit("No valid records in the file.")

    print(f"Total records: {len(bleu_list)}")
    print(f"BLEU   : {statistics.mean(bleu_list):.4f}")
    print(f"ROUGE-L: {statistics.mean(rouge_list):.4f}")
    print(f"METEOR : {statistics.mean(meteor_list):.4f}")


if __name__ == "__main__":
    main()
