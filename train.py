"""Train and test DAVIN.

  S_cg(d, i) = cos(h_d, h_i)
  S_fg(d, i) = Σ_n Σ_m α_nm · cos(h_{p_n}, h_{a_m})
  S(d, i)    = λ · S_cg(d, i) + (1 − λ) · S_fg(d, i),   λ = σ(θ) learned
  L          = InfoNCE(S, τ) over the target item and K hard negatives sampled from softmax(S / γ)

The dialogue encoder (LoRA), the joint attention component, and λ are trained; the item encoder is frozen.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, get_linear_schedule_with_warmup

from davin.attention import JointAttention
from davin.data import DialogueDataset, flatten_preferences, load_catalog, load_item_attributes, read_json
from davin.encoding import encode_texts, preserve_rng
from davin.item_bank import load_item_bank
from davin.matching import MultiGranularMatching
from davin.sampling import NegativeSampler, PositiveSafeBatchSampler

METRICS = ('H@1', 'R@1', 'N@10', 'M@10')
SELECTION_METRIC = 'N@10'  # The best epoch on VALID, also used for early stopping


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main_category', required=True)
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--valid_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--item_attr_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--item_data_path', default='data/item/metadata.json')
    parser.add_argument('--images_dirs', nargs='+', default=['data/item/images'])
    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-Embedding-8B')
    parser.add_argument('--model_revision', default='2c4565515e0f265c6511776e7193b22c0968ddc7')

    parser.add_argument('--lambda_init', type=float, default=0.5, help='Initial λ, the weight of S_cg')
    parser.add_argument('--lambda_lr', type=float, default=0.02)
    parser.add_argument('--tau', type=float, default=0.0333, help='InfoNCE temperature')
    parser.add_argument('--gamma', type=float, default=0.0333, help='Hard negative sampling temperature')
    parser.add_argument('--d_k', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_negatives', type=int, default=31, help='K hard negatives per dialogue')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of the dialogue encoder')
    parser.add_argument('--attention_lr', type=float, default=5e-4, help='Learning rate of the joint attention component')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--encode_batch_size', type=int, default=8)
    parser.add_argument('--attr_batch_size', type=int, default=32)
    parser.add_argument('--mining_chunk_size', type=int, default=256)
    return parser.parse_args()


def load_model(args):
    processor = AutoProcessor.from_pretrained(args.model_name, revision=args.model_revision)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, revision=args.model_revision, torch_dtype=torch.bfloat16,
                                                            device_map='auto', quantization_config=quantization)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    return model, processor


@torch.no_grad()
def score_catalog(args, matching, catalog, bank, h_d, preferences):
    chunks = []
    for start in range(0, len(catalog), args.mining_chunk_size):
        chunk = catalog[start:start + args.mining_chunk_size]
        chunks.append(matching(h_d, preferences, bank.select(chunk), len(h_d), len(chunk)).float().cpu())
    return torch.cat(chunks, 1)


@torch.no_grad()
def evaluate(args, model, processor, matching, samples, catalog, bank, device, hidden_size):
    """Rank the whole catalog for each dialogue."""
    model.eval(); matching.eval()
    torch.cuda.empty_cache()
    catalog_ids = [item['item_id'] for item in catalog]
    items = bank.select(catalog)
    totals = dict.fromkeys(METRICS, 0.0)
    for start in range(0, len(samples), args.encode_batch_size):
        batch = samples[start:start + args.encode_batch_size]
        h_d = encode_texts(model, processor.tokenizer, [s['dialogue'] for s in batch], device)
        texts, pref_owner = flatten_preferences(batch, device)
        pref_emb = encode_texts(model, processor.tokenizer, texts, device) if texts else torch.zeros(0, hidden_size, device=device)
        scores = matching(h_d, (pref_emb, pref_owner), items, len(batch), len(catalog)).cpu().numpy()
        order = np.argsort(scores, axis=1)[:, ::-1][:, :10]
        for row, sample in enumerate(batch):
            targets = set(sample['target_items'])
            relevant = [catalog_ids[i] in targets for i in order[row]]
            totals['H@1'] += float(relevant[0])
            totals['R@1'] += relevant[0] / len(targets)
            dcg = sum(1.0 / math.log2(rank + 2) for rank, hit in enumerate(relevant) if hit)
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(10, len(targets))))
            totals['N@10'] += dcg / ideal
            totals['M@10'] += 1.0 / (relevant.index(True) + 1) if any(relevant) else 0.0
    return {key: value / len(samples) for key, value in totals.items()}


def format_metrics(metrics):
    return ' | '.join(f'{key} {metrics[key]:.4f}' for key in METRICS)


def identity_collate(batch):
    return batch


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    if (output / 'config.json').exists():
        raise SystemExit(f'{output} already holds a run')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    logger.add(output / 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {message}', level='INFO')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metadata = read_json(args.item_data_path)
    attr_map = load_item_attributes(args.item_attr_path)
    train_set = DialogueDataset(args.train_path, metadata, args.images_dirs, attr_map, expand_gt=True)
    valid_samples = DialogueDataset(args.valid_path, metadata, args.images_dirs, attr_map).samples
    test_samples = DialogueDataset(args.test_path, metadata, args.images_dirs, attr_map).samples
    catalog = load_catalog(metadata, args.main_category, args.images_dirs, attr_map)
    logger.info(f'{len(train_set)} training samples | {len(valid_samples)} valid | {len(test_samples)} test | {len(catalog)} catalog items')

    model, processor = load_model(args)
    hidden_size = model.config.text_config.hidden_size
    bank = load_item_bank(args, model, processor, catalog, train_set.samples, device, hidden_size)

    with preserve_rng():
        joint_attention = JointAttention(hidden_size, d_k=args.d_k).to(device)
        matching = MultiGranularMatching(joint_attention, args.lambda_init)

    def save_best():
        adapter_dir = output / 'adapter_best'
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        torch.save(dict(joint_attention_config=joint_attention.get_config(), lambda_init=args.lambda_init,
                        state={k: v.detach().cpu() for k, v in matching.state_dict().items()}),
                   output / 'matching_best.pt')

    batch_plan = PositiveSafeBatchSampler(train_set.samples, args.batch_size, args.seed, args.num_epochs)
    train_loader = DataLoader(train_set, batch_sampler=batch_plan, collate_fn=identity_collate, num_workers=2)
    sampler = NegativeSampler(catalog, args.seed, args.num_negatives, args.gamma)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    matching_params = list(matching.parameters())
    optimizer = torch.optim.AdamW([dict(params=lora_params, lr=args.lr, weight_decay=0.01),
                                   dict(params=list(joint_attention.parameters()), lr=args.attention_lr, weight_decay=0.0),
                                   dict(params=[matching.theta], lr=args.lambda_lr, weight_decay=0.0)])
    total_steps = sum(len(plan) for plan in batch_plan.plans)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best = dict(value=0.0, epoch=-1, lora=None, matching=None)
    stale_epochs = 0
    for epoch in range(1, args.num_epochs + 1):
        batch_plan.set_epoch(epoch - 1)
        model.train(); matching.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', mininterval=1.0):
            n_dialogues = len(batch)
            texts, pref_owner = flatten_preferences(batch, device)
            embeddings = encode_texts(model, processor.tokenizer, [s['dialogue'] for s in batch] + texts, device)
            h_d, pref_emb = embeddings[:n_dialogues], embeddings[n_dialogues:]

            catalog_scores = score_catalog(args, matching, catalog, bank, h_d.detach(), (pref_emb.detach(), pref_owner))
            batch = sampler.sample(batch, catalog_scores)

            # Each dialogue sees only its target item and its own hard negatives.
            candidates, column = [], {}
            for sample in batch:
                if sample['item_id'] not in column:
                    column[sample['item_id']] = len(candidates)
                    candidates.append(sample)
            for sample in batch:
                for index in sample['negatives']:
                    if catalog[index]['item_id'] not in column:
                        column[catalog[index]['item_id']] = len(candidates)
                        candidates.append(catalog[index])
            labels = torch.tensor([column[s['item_id']] for s in batch], dtype=torch.long, device=device)
            hidden = torch.ones(n_dialogues, len(candidates), dtype=torch.bool, device=device)
            for row, sample in enumerate(batch):
                hidden[row, [column[catalog[i]['item_id']] for i in sample['negatives']]] = False
                hidden[row, labels[row]] = False

            scores = matching(h_d, (pref_emb, pref_owner), bank.select(candidates), n_dialogues, len(candidates))
            loss = F.cross_entropy(scores.masked_fill(hidden, float('-inf')) / args.tau, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in epoch {epoch}')
            loss.backward()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(lora_params + matching_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        epoch_loss /= max(1, len(train_loader))
        valid = evaluate(args, model, processor, matching, valid_samples, catalog, bank, device, hidden_size)
        lam = float(matching.lam)
        logger.info(f'Epoch {epoch} | loss {epoch_loss:.4f} | λ {lam:.4f} | VALID {format_metrics(valid)}')
        with open(output / 'epochs.jsonl', 'a') as stream:
            stream.write(json.dumps(dict(epoch=epoch, loss=epoch_loss, lam=lam, valid=valid)) + '\n')

        if best['lora'] is None or valid[SELECTION_METRIC] > best['value']:
            best.update(value=valid[SELECTION_METRIC], epoch=epoch,
                        lora={k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k.lower()},
                        matching=[p.detach().clone() for p in matching_params])
            save_best()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            logger.info('Early stopping')
            break

    state = model.state_dict()
    state.update({k: v.to(state[k].device) for k, v in best['lora'].items()})
    model.load_state_dict(state, strict=False)
    for parameter, value in zip(matching_params, best['matching']):
        parameter.data.copy_(value)
    test = evaluate(args, model, processor, matching, test_samples, catalog, bank, device, hidden_size)
    lam = float(matching.lam)
    logger.info(f'Best VALID {SELECTION_METRIC} (epoch {best["epoch"]}, λ {lam:.4f}) | TEST {format_metrics(test)}')
    (output / 'test_metrics.json').write_text(json.dumps(dict(selected_by=f'VALID {SELECTION_METRIC}', epoch=best['epoch'],
                                                              lam=lam, **test), indent=2))


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda message: tqdm.write(message, end=''), format='{time:YYYY-MM-DD HH:mm:ss} | {message}', level='INFO')
    main(parse_args())
