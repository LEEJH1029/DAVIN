"""Item and attribute embeddings from the frozen item encoder, cached on disk."""
import hashlib
import json
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from davin.encoding import encode_items, encode_texts, pretrained_encoder


class ItemBank:
    def __init__(self, items, item_embeddings, attribute_embeddings, device):
        self.items = items
        self.index = {item['item_id']: i for i, item in enumerate(items)}
        self.item_embeddings = item_embeddings.float().to(device)
        self.attribute_embeddings = attribute_embeddings.float().to(device)
        self.offsets = [0]
        for item in items:
            self.offsets.append(self.offsets[-1] + len(item['attr_texts']))

    def select(self, samples):
        """Return (h_i, h_{a_m}, the owning row of each attribute)."""
        indices, attr_indices, owner = [], [], []
        for row, sample in enumerate(samples):
            index = self.index[sample['item_id']]
            indices.append(index)
            rows = range(self.offsets[index], self.offsets[index + 1])
            attr_indices.extend(rows)
            owner.extend([row] * len(rows))
        device = self.item_embeddings.device
        return (self.item_embeddings.index_select(0, torch.tensor(indices, dtype=torch.long, device=device)),
                self.attribute_embeddings.index_select(0, torch.tensor(attr_indices, dtype=torch.long, device=device)),
                torch.tensor(owner, dtype=torch.long, device=device))


def _encode(args, model, processor, items, device, hidden_size):
    texts = [text for item in items for text in item['attr_texts']]
    item_embeddings, attribute_embeddings = [], []
    with pretrained_encoder(model):
        for start in tqdm(range(0, len(items), args.encode_batch_size), desc='Encoding items'):
            item_embeddings.append(encode_items(model, processor, items[start:start + args.encode_batch_size], device).cpu())
        for start in tqdm(range(0, len(texts), args.attr_batch_size), desc='Encoding attributes'):
            attribute_embeddings.append(encode_texts(model, processor.tokenizer, texts[start:start + args.attr_batch_size], device).cpu())
    return dict(item_embeddings=torch.cat(item_embeddings),
                attribute_embeddings=torch.cat(attribute_embeddings) if attribute_embeddings else torch.empty(0, hidden_size))


def load_item_bank(args, model, processor, catalog, train_samples, device, hidden_size):
    """Cover the catalog and the training target items outside it."""
    items, seen = [], set()
    for sample in list(catalog) + list(train_samples):
        if sample['item_id'] not in seen:
            seen.add(sample['item_id'])
            items.append({key: sample[key] for key in ('item_id', 'title', 'image', 'attr_texts')})

    key = hashlib.sha256(json.dumps(dict(model=args.model_name, revision=args.model_revision, encode_batch_size=args.encode_batch_size,
                                         attr_batch_size=args.attr_batch_size, items=items),
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
    path = Path(args.cache_dir) / f'items_{key}.pt'
    if path.exists():
        logger.info(f'Reusing {path}')
        payload = torch.load(path, map_location='cpu', weights_only=True)
    else:
        payload = _encode(args, model, processor, items, device, hidden_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    return ItemBank(items, payload['item_embeddings'], payload['attribute_embeddings'], device)
