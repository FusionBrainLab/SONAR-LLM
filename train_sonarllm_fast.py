import os
import time
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

import wandb

import concurrent.futures

from datasets import load_dataset

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline
)

from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM
)

from transformers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


# ------------------------------------------------------------------------
# 1) Parse command line arguments
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=820, help="Warmup steps for the LR scheduler")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Use fewer samples for quick debugging. -1 for all.")
    parser.add_argument("--max_val_samples", type=int, default=2048,
                        help="Use fewer samples for quick debugging. -1 for all.")
    parser.add_argument("--output_dir", type=str, default="./ddp_checkpoints_2")

    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="my_project", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default="my_ddp_run", help="Wandb run name.")

    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Use mixed precision training (fp16 autocast).")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")

    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (use_cache=False).")

    parser.add_argument("--start_from", type=str, default=None,
                        help="Path to a full checkpoint from which to resume training (including mid-epoch).")

    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------
# 2) Dataset classes and data collator
# ------------------------------------------------------------------------
class NextSentenceEmbeddingDataset(Dataset):
    def __init__(self, items, max_sentences=128):
        self.texts = []
        self.embeddings = []
        self.max_sentences = max_sentences
        self.sonar_eos_embedding = torch.load('./sonar_embeddings_ts_100_shuffle/sonar_eos_embedding.pt')
        
        for item in items:
            text_embs = item['embeddings'][:max_sentences]
            text_embs.append(self.sonar_eos_embedding)
            sentences = item['sentences'][:max_sentences]
            sentences.append("End of sequence.")
            if len(text_embs) >= 2:
                self.embeddings.append(text_embs)
                self.texts.append(sentences)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_embs = self.embeddings[idx]
        sentences = self.texts[idx]
        item = {
            "embeddings": torch.tensor(text_embs, dtype=torch.float32),
            "sentences": sentences
        }
        return item


def data_collator(batch):
    max_len = max(len(item["embeddings"]) for item in batch)
    B = len(batch)

    embed_batch = []
    text_batch = []
    seq_lens = []

    for item in batch:
        embs = item["embeddings"]
        txts = item["sentences"]
        seq_len = len(embs)

        padded_embs = []
        padded_txts = []
        for i in range(max_len):
            if i < seq_len:
                padded_embs.append(embs[i].unsqueeze(0))
                padded_txts.append(txts[i])
            else:
                padded_embs.append(torch.zeros((1, 1024)))
                padded_txts.append("")
        padded_embs = torch.cat(padded_embs, dim=0)
        embed_batch.append(padded_embs)
        text_batch.append(padded_txts)
        seq_lens.append(seq_len)

    embed_batch = torch.stack(embed_batch, dim=0)
    return {
        "embeddings": embed_batch,
        "texts": text_batch,
        "seq_lens": seq_lens
    }


# ------------------------------------------------------------------------
# 3) Projector and SonarLossWrapper
# ------------------------------------------------------------------------
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)


class SonarLossWrapper(nn.Module):
    def __init__(self, llama_model, forward_proj, reverse_proj, sonar_decoder):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_decoder = sonar_decoder

        for p in self.sonar_decoder.parameters():
            p.requires_grad = False

    def forward(self, embeddings_1024, texts, seq_lens):
        device = embeddings_1024.device
        B, T, _ = embeddings_1024.shape

        embs_projected = self.forward_proj(embeddings_1024)
        llama_out = self.llama_model(
            inputs_embeds=embs_projected,
            output_hidden_states=True
        )
        last_hidden = llama_out.hidden_states[-1]

        pred_hidden_list = []
        ref_texts_list = []
        for b in range(B):
            seqlen = seq_lens[b]
            for k in range(1, seqlen):
                pred_hidden_list.append(last_hidden[b, k - 1, :])
                ref_texts_list.append(texts[b][k])

        if len(pred_hidden_list) == 0:
            return torch.tensor(0.0, device=device)

        pred_hidden_batch = torch.stack(pred_hidden_list, dim=0)
        pred_emb_1024 = self.reverse_proj(pred_hidden_batch)

        with torch.no_grad():
            target_text_encoder = self.sonar_decoder.tokenizer.create_encoder(
                task="translation", lang="eng_Latn", mode="target", device=device
            )
        encoded_texts = [target_text_encoder(t) for t in ref_texts_list]
        lengths = [et.size(0) for et in encoded_texts]
        max_len = min(max(lengths), 256)

        pad_idx = self.sonar_decoder.tokenizer.vocab_info.pad_idx
        dec_ids = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        labels = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        for i, et in enumerate(encoded_texts):
            dec_ids[i, : min(len(et), max_len)] = et[:max_len]
            et = torch.cat([et[1:], torch.tensor([3]).to(device)])
            labels[i, : min(len(et), max_len)] = et[:max_len]

        enc_output = pred_emb_1024.unsqueeze(1)
        dec_out, dec_pad_mask = self.sonar_decoder.model.decode(
            seqs=dec_ids,
            padding_mask=None,
            encoder_output=enc_output,
            encoder_padding_mask=None
        )
        final_out = self.sonar_decoder.model.project(dec_out, dec_pad_mask)
        logits = final_out.logits

        vocab_size = logits.size(-1)
        logits_2d = logits.view(-1, vocab_size)
        labels_1d = labels.view(-1)

        ce_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
        total_ce = ce_fn(logits_2d, labels_1d)

        return total_ce


def load_file(i):
    data = torch.load(f'./sonar_embeddings_ts_100_shuffle/tiny_stories_sonar_embeddings_train_bucket_{i}.pt')
    return data['train']


# ------------------------------------------------------------------------
# 4) Main Training Script
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    # --------------------------------------------------------------------
    # 4a) Distributed setup
    # --------------------------------------------------------------------
    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl", timeout=timedelta(minutes=40))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    is_main_process = (not distributed) or (dist.get_rank() == 0)
    if is_main_process:
        if distributed:
            print(f"Using {world_size} GPUs for training.")
        else:
            print("Single-process training on device:", device)
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            wandb.config.update(vars(args))

    # --------------------------------------------------------------------
    # 4b) Model config + optional gradient checkpointing
    # --------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    configuration = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=10,
        num_attention_heads=16,
        hidden_act='silu',
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=64,
    )

    if args.gradient_checkpointing:
        configuration.use_cache = False
        configuration.gradient_checkpointing = True

    llama_model = LlamaForCausalLM(configuration).to(device)
    if args.gradient_checkpointing:
        llama_model.gradient_checkpointing_enable()

    # --------------------------------------------------------------------
    # 4c) Sonar pipelines (frozen)
    # --------------------------------------------------------------------
    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    ).eval()
    for param in vec2text_model.parameters():
        param.requires_grad = False

    # --------------------------------------------------------------------
    # 4d) Projectors + Wrapper
    # --------------------------------------------------------------------
    forward_projector = Projector(1024, 512).to(device)
    reverse_projector = Projector(512, 1024).to(device)
    model = SonarLossWrapper(
        llama_model=llama_model,
        forward_proj=forward_projector,
        reverse_proj=reverse_projector,
        sonar_decoder=vec2text_model
    ).to(device)

    # --------------------------------------------------------------------
    # 4e) Load validation data (once)
    # --------------------------------------------------------------------
    if is_main_process:
        print("Loading validation data")
    val_items = torch.load('./sonar_embeddings_ts_100_shuffle/tiny_stories_sonar_embeddings_val.pt')
    if is_main_process:
        print("Validation data ready")
    val_key = 'validation' if 'validation' in val_items else 'val'
    val_dataset = NextSentenceEmbeddingDataset(val_items[val_key], max_sentences=32)
    if distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None
    val_dloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=data_collator,
        drop_last=False
    )

    # --------------------------------------------------------------------
    # 4f) Optimizer & Scheduler
    # --------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=16480 #4 epochs, global_bs=512
    )

    scaler = GradScaler(enabled=args.use_mixed_precision)

    # --------------------------------------------------------------------
    # 4g) Optionally resume from a checkpoint (mid-epoch)
    # --------------------------------------------------------------------
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0

    if args.start_from is not None:
        if is_main_process:
            print(f"Loading checkpoint from {args.start_from}")
        ckpt_data = load_full_checkpoint(args.start_from, model, optimizer, scheduler, device)
        start_epoch = ckpt_data["epoch"]
        start_step_in_epoch = ckpt_data["step_in_epoch"]
        global_step = ckpt_data["global_step"]

    # --------------------------------------------------------------------
    # 4h) Wrap in DDP if needed
    # --------------------------------------------------------------------
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # --------------------------------------------------------------------
    # 4i) Training Loop with segmented bucket loading (5 buckets per segment)
    # --------------------------------------------------------------------
    model.train()
    full_loss = 0.0
    accumulated_steps = 0.0
    total_fwd_passes = 0

    for epoch in range(start_epoch, args.epochs):
        if is_main_process:
            print(f"Starting epoch {epoch+1}")

        # Process buckets in segments of 5 (i.e. 0-4, 5-9, ..., 95-99)
        for bucket_start in range(0, 100, 5):
            bucket_end = bucket_start + 5
            if is_main_process:
                print(f"Loading training buckets {bucket_start} to {bucket_end - 1}")
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                parts = list(executor.map(load_file, range(bucket_start, bucket_end)))
            train_items = []
            for part in parts:
                train_items.extend(part)
            if is_main_process:
                print(f"Buckets {bucket_start}-{bucket_end - 1} loaded, total items: {len(train_items)}")
            
            # Create a new dataset and DataLoader for this segment
            train_dataset = NextSentenceEmbeddingDataset(train_items, max_sentences=32)
            if distributed:
                train_sampler = DistributedSampler(train_dataset)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler = None
            train_dloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=0,
                collate_fn=data_collator,
                drop_last=False
            )

            # Iterate over the DataLoader batches for the current segment
            for step, batch in enumerate(tqdm(train_dloader, desc=f"Epoch {epoch+1} - Buckets {bucket_start}-{bucket_end - 1}")):
                # Skip steps if resuming mid-epoch (only for the first segment)
                total_fwd_passes += 1
                if epoch == start_epoch and step < start_step_in_epoch:
                    continue

                embeddings_1024 = batch["embeddings"].to(device)
                texts = batch["texts"]
                seq_lens = batch["seq_lens"]

                with autocast(enabled=args.use_mixed_precision, dtype=torch.float16):
                    loss = model(embeddings_1024, texts, seq_lens)
                    loss = loss / args.grad_accum_steps
                    full_loss += loss.item()

                scaler.scale(loss).backward()
                accumulated_steps += 1

                if (step + 1) % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (global_step % args.logging_steps == 0) and is_main_process:
                        current_lr = scheduler.get_last_lr()[0]
                        avg_loss = args.grad_accum_steps * full_loss / accumulated_steps
                        print(f"[Epoch {epoch+1}, Global Step {global_step}] Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")
                        
                        if args.use_wandb:
                            samples_seen = total_fwd_passes * args.batch_size
                            if distributed:
                                samples_seen *= world_size
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/step": global_step,
                                "train/samples_seen": samples_seen,
                                "train/lr": current_lr,
                            })
                    full_loss = 0.0
                    accumulated_steps = 0.0

                    if (global_step % args.eval_steps == 0):
                        val_loss = evaluate(model, val_dloader, device, distributed)
                        if is_main_process:
                            print(f"[Eval] Global Step {global_step}, Val Loss = {val_loss:.4f}")
                            if args.use_wandb:
                                wandb.log({"eval/loss": val_loss, "eval/step": global_step})
                    
            del train_dloader
            del train_sampler
            del train_dataset
            del train_items
            del parts
            
        # End of epoch: Save checkpoint if desired. 
        val_loss_epoch = evaluate(model, val_dloader, device, distributed)
        if is_main_process:
            print(f"[Epoch {epoch+1}] Validation Loss = {val_loss_epoch:.4f}")
            if args.use_wandb:
                wandb.log({"eval/loss_epoch": val_loss_epoch, "eval/epoch": epoch + 1})

        if is_main_process:
            save_full_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step_in_epoch=0,
                global_step=global_step,
                output_dir=args.output_dir
            )

    if distributed:
        dist.barrier()


# ------------------------------------------------------------------------
# 5) Evaluation function
# ------------------------------------------------------------------------
def evaluate(model, val_dloader, device, distributed):
    model.eval()

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in val_dloader:
            embeddings_1024 = batch["embeddings"].to(device)
            texts = batch["texts"]
            seq_lens = batch["seq_lens"]

            with autocast(enabled=False):
                loss = model(embeddings_1024, texts, seq_lens)

            bs = embeddings_1024.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    if distributed:
        result = torch.tensor([total_loss, total_count], device=device, dtype=torch.float32)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        total_loss, total_count = result[0].item(), result[1].item()

    model.train()
    return (total_loss / total_count) if total_count > 0 else 0.0


# ------------------------------------------------------------------------
# 6) Full-checkpoint Save & Load
# ------------------------------------------------------------------------
def save_full_checkpoint(model,
                         optimizer,
                         scheduler,
                         epoch,
                         step_in_epoch,
                         global_step,
                         output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")

    checkpoint_dict = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "random_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }

    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Saved full checkpoint to {checkpoint_path}")


def load_full_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model

    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    random.setstate(ckpt["random_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if "cuda_rng_state_all" in ckpt:
        for i, rng_state in enumerate(ckpt["cuda_rng_state_all"]):
            torch.cuda.set_rng_state(rng_state.cpu(), device=i)

    return {
        "epoch": ckpt["epoch"],
        "step_in_epoch": ckpt["step_in_epoch"],
        "global_step": ckpt["global_step"],
    }


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
