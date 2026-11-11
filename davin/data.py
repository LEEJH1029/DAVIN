import json
import os
import re
from glob import glob

import torch
from torch.utils.data import Dataset


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def sanitize(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')


def _category(entry: dict) -> str:
    return (entry.get('category') or '').strip()


def entry_text(entry: dict) -> str:
    """The text of a preference or an attribute."""
    return f"{sanitize(_category(entry))} : {sanitize((entry.get('value') or '').strip())}"


def item_title_text(info: dict, max_chars: int = 2048) -> str:
    title = (info.get('title') or '').strip()
    return f'Title: {sanitize(title)}'[:max_chars] if title else 'Title:'


def find_main_image(images_dirs, item_id):
    for directory in images_dirs:
        path = os.path.join(directory, f'{item_id}_0.jpg')
        if os.path.exists(path):
            return path
    for directory in images_dirs:
        candidates = sorted(glob(os.path.join(directory, f'{item_id}_*.jpg')))
        if candidates:
            return candidates[0]
    return None


def load_item_attributes(path):
    return {item_id: record['merged_attributes'] for item_id, record in read_json(path).items()
            if isinstance(record.get('merged_attributes'), list)}


def item_attributes(attr_map, item_id):
    attributes = [a for a in attr_map.get(item_id, []) if isinstance(a, dict)]
    return [entry_text(a) for a in attributes]


def load_catalog(metadata, main_category, images_dirs, attr_map):
    """The domain items that have a representative image."""
    catalog = []
    for item_id, info in metadata.items():
        if info.get('main_category') != main_category:
            continue
        image = find_main_image(images_dirs, item_id)
        if image is not None:
            catalog.append(dict(item_id=item_id, title=item_title_text(info), image=image, attr_texts=item_attributes(attr_map, item_id)))
    return catalog


class DialogueDataset(Dataset):
    """(dialogue, target item) pairs: one per target item if expand_gt, otherwise only the first."""

    def __init__(self, path, metadata, images_dirs, attr_map, expand_gt=False):
        self.samples = []
        for record in read_json(path):
            targets = record.get('gt_items') or []
            valid = [t for t in targets if t in metadata and find_main_image(images_dirs, t) is not None]
            if not valid:
                continue
            preferences = [p for p in (record.get('preferences') or []) if _category(p) or p.get('value')]
            base = dict(dialogue=sanitize(record.get('context') or '')[:4096], target_items=targets,
                        pref_texts=[entry_text(p) for p in preferences])
            for item_id in (valid if expand_gt else valid[:1]):
                self.samples.append(dict(base, item_id=item_id, image=find_main_image(images_dirs, item_id),
                                         title=item_title_text(metadata[item_id]), attr_texts=item_attributes(attr_map, item_id)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def flatten_preferences(samples, device):
    """Return (texts, index of the owning dialogue) of a batch."""
    texts, owner = [], []
    for index, sample in enumerate(samples):
        texts.extend(sample['pref_texts']); owner.extend([index] * len(sample['pref_texts']))
    return texts, torch.tensor(owner, dtype=torch.long, device=device)
