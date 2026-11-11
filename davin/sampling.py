import random

import torch


class NegativeSampler:
    """Draws negatives without replacement from softmax(S / γ) over the catalog, excluding the target items of the dialogue."""

    def __init__(self, catalog, seed, count, temperature):
        self.catalog_ids = [item['item_id'] for item in catalog]
        self.count, self.temperature = count, temperature
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed) + 104729)

    def _gumbel_top_k(self, log_weights):
        uniform = torch.rand(len(log_weights), dtype=torch.float64, generator=self.generator)
        uniform = uniform.clamp(min=torch.finfo(torch.float64).tiny, max=1 - torch.finfo(torch.float64).eps)
        return torch.topk(log_weights - torch.log(-torch.log(uniform)), self.count, sorted=True).indices

    @torch.no_grad()
    def sample(self, batch, scores):
        scores = scores.double()
        out = []
        for row, sample in enumerate(batch):
            banned = set(sample['target_items']) | {sample['item_id']}
            eligible = torch.tensor([i for i, item_id in enumerate(self.catalog_ids) if item_id not in banned], dtype=torch.long)
            log_weights = scores[row][eligible] / self.temperature
            log_weights -= log_weights.max()
            out.append(dict(sample, negatives=eligible[self._gumbel_top_k(log_weights)].tolist()))
        return out


class PositiveSafeBatchSampler:
    """Shuffles each epoch and defers samples whose target items collide with the current batch."""

    def __init__(self, samples, batch_size, seed, epochs):
        self.epoch = 0
        self.plans = []
        targets = [set(sample['target_items']) | {sample['item_id']} for sample in samples]
        for epoch in range(epochs):
            pending = list(range(len(samples)))
            random.Random(seed + epoch).shuffle(pending)
            batches = []
            while len(pending) >= batch_size:
                batch, deferred, used_items, used_targets = [], [], set(), set()
                for index in pending:
                    item_id = samples[index]['item_id']
                    if len(batch) < batch_size and item_id not in used_targets and not targets[index] & used_items:
                        batch.append(index)
                        used_items.add(item_id)
                        used_targets.update(targets[index])
                    else:
                        deferred.append(index)
                if len(batch) < batch_size:
                    break
                batches.append(batch)
                pending = deferred
            if not batches:
                raise ValueError('Cannot form a full positive-safe batch')
            self.plans.append(batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.plans[self.epoch])

    def __len__(self):
        return len(self.plans[self.epoch])
