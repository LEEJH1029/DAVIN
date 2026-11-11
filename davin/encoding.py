from contextlib import contextmanager
import random

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def preserve_rng():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def pretrained_encoder(model):
    """Encode with the frozen pretrained model, leaving the training state and RNG untouched."""
    modes = {module: module.training for module in model.modules()}
    try:
        with preserve_rng(), torch.no_grad(), model.disable_adapter():
            model.eval()
            yield
    finally:
        for module, training in modes.items():
            module.training = training


def _trunk(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return base.model


def _last_token_embedding(hidden, attention_mask):
    last = attention_mask.size(1) - 1 - attention_mask.flip(1).argmax(1)
    return F.normalize(hidden[torch.arange(hidden.size(0), device=hidden.device), last].float(), dim=-1)


def encode_texts(model, tokenizer, texts, device, batch_chars=32768):
    """Encode texts in length-sorted batches of at most batch_chars padded characters."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    start = 0
    while start < len(texts):
        end, longest = start, 0
        while end < len(texts):
            candidate = max(longest, len(texts[order[end]]))
            if end > start and (end - start + 1) * candidate > batch_chars:
                break
            longest, end = candidate, end + 1
        indices = order[start:end]
        batch = tokenizer([texts[i] for i in indices], padding=True, truncation=True, max_length=2048,
                          return_tensors='pt').to(device)
        hidden = _trunk(model)(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               return_dict=True).last_hidden_state
        embeddings = _last_token_embedding(hidden, batch['attention_mask'])
        for position, index in enumerate(indices):
            out[index] = embeddings[position]
        start = end
    return torch.stack(out, 0)


def encode_items(model, processor, rows, device):
    """Encode items from their representative image and title."""
    images = []
    for row in rows:
        with Image.open(row['image']) as source:
            images.append(source.convert('RGB'))
    try:
        prompts = [processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['title']}]}],
            tokenize=False, add_generation_prompt=False) for row in rows]
        batch = processor(text=prompts, padding=True, return_tensors='pt', images=images).to(device)
        hidden = _trunk(model)(**batch, return_dict=True).last_hidden_state
        return _last_token_embedding(hidden, batch['attention_mask'])
    finally:
        for image in images:
            image.close()
