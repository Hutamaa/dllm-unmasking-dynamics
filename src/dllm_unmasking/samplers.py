"""Shared irreversible LLaDA sampler; only position ordering varies."""
import random
import torch

STRATEGIES = ('random', 'left_to_right', 'right_to_left', 'confidence', 'influence')


def influence_predictions(model, x, start, mask_id):
    """Response predictions and entropy in nats; MASK excluded from vocabulary."""
    output = model(x, use_cache=False)
    logits = output.logits[0, start:].float().clone()
    del output
    logits[:, mask_id] = -torch.inf
    predictions = logits.argmax(dim=-1)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    confidence = log_probs.gather(1, predictions[:, None]).squeeze(1).exp()
    probabilities = log_probs.exp()
    log_probs[:, mask_id] = 0  # avoid 0 * -inf in entropy
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return predictions, confidence, entropy


def influence_reveal(model, x, start, revealed, lo, hi, k, mask_id, top_k):
    """Sequential greedy counterfactual selection, updating baseline after each pick."""
    picks, tokens, confidences, scores, rounds = [], [], [], [], []
    passes = 0
    for _ in range(k):
        predictions, confidence, entropy = influence_predictions(model, x, start, mask_id)
        passes += 1
        eligible = ~revealed[lo:hi]
        # Explicit -inf exclusion; stable confidence ranking defines a top-K pool.
        ranking = confidence[lo:hi].masked_fill(~eligible, -torch.inf)
        pool_size = int(eligible.sum().item())
        if top_k is not None:
            pool_size = min(pool_size, top_k)
        candidates = torch.argsort(ranking, descending=True, stable=True)[:pool_size] + lo
        # Sort by position so equal influence scores favor the lower position.
        candidates = candidates.sort().values
        influence = torch.full_like(confidence, -torch.inf)
        masked = ~revealed  # includes masked positions in future blocks
        for position in candidates.cpu().tolist():
            other = masked.clone()
            other[position] = False
            # Clone the original state for every probe; probes cannot mutate x.
            probe = x.clone()
            probe[0, start + position] = predictions[position]
            _, _, counterfactual_entropy = influence_predictions(model, probe, start, mask_id)
            passes += 1
            influence[position] = (entropy[other] - counterfactual_entropy[other]).sum()
        selected = influence.argmax()  # unrevealed noncandidates remain -inf
        position = int(selected.item())
        token = predictions[position]
        picks.append(selected.clone())
        tokens.append(token.clone())
        confidences.append(confidence[position].clone())
        scores.append(influence[position].clone())
        rounds.append(dict(candidate_positions=candidates.clone(),
                           candidate_scores=influence[candidates].clone()))
        x[0, start + position] = token
        revealed[position] = True
    return (torch.stack(picks), torch.stack(tokens), torch.stack(confidences),
            torch.stack(scores), rounds, passes)


def transfer_schedule(gen_length, steps, block_length):
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError('gen_length must be a positive multiple of block_length')
    blocks = gen_length // block_length
    if steps <= 0 or steps % blocks or steps > gen_length:
        raise ValueError('steps must divide across blocks and be <= gen_length')
    per_block = steps // blocks
    base, remainder = divmod(block_length, per_block)
    return [base + (i < remainder) for i in range(per_block)]


@torch.inference_mode()
def generate(model, prompt, strategy, *, gen_length, steps, block_length,
             mask_id=126336, seed=42, influence_top_k=None):
    """Batch size one. Return response IDs, GPU seconds, wall seconds, trajectory.

    Confidence is softmax probability after suppressing MASK as an output token.
    EOS does not stop decoding: every strategy performs the same number of steps.
    CPU trajectory transfers happen after timing; positions are response-relative.
    """
    import time
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if influence_top_k is not None and (type(influence_top_k) is not int or influence_top_k <= 0):
        raise ValueError('influence_top_k must be null (exhaustive) or a positive integer')
    schedule = transfer_schedule(gen_length, steps, block_length)
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError('Expected one unpadded prompt')
    device = prompt.device
    rng = random.Random(seed)  # independent of global/model RNG
    start = prompt.shape[1]
    x = torch.full((1, start + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :start] = prompt
    revealed = torch.zeros(gen_length, dtype=torch.bool, device=device)
    trace = []
    gpu = device.type == 'cuda'
    if gpu:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    if gpu:
        begin.record()
    for block in range(gen_length // block_length):
        lo, hi = block * block_length, (block + 1) * block_length
        # CPU list avoids CUDA synchronization when identifying eligible positions.
        remaining = list(range(lo, hi))
        for k in schedule:
            if strategy == 'influence':
                positions, tokens, selected_confidence, influence, rounds, passes = influence_reveal(
                    model, x, start, revealed, lo, hi, k, mask_id, influence_top_k)
                trace.append(dict(block=block, positions=positions, token_ids=tokens,
                                  confidences=selected_confidence, influence_scores=influence,
                                  greedy_rounds=rounds, forward_passes=passes,
                                  influence_top_k=influence_top_k,
                                  influence_approximation=('exhaustive' if influence_top_k is None
                                                           else 'confidence_top_k')))
                continue
            output = model(x, use_cache=False)
            logits = output.logits[0, start + lo:start + hi].float().clone()
            del output
            logits[:, mask_id] = -torch.inf  # a revealed token must not remain MASK
            predictions = logits.argmax(dim=-1)
            confidence = torch.exp(logits.gather(1, predictions[:, None]).squeeze(1)
                                   - torch.logsumexp(logits, dim=-1))
            if strategy == 'confidence':
                eligible = ~revealed[lo:hi]
                scores = confidence.masked_fill(~eligible, -torch.inf)
                # Stable ranking breaks confidence ties by lower position.
                local = torch.argsort(scores, descending=True, stable=True)[:k]
                positions = local + lo
            else:
                if strategy == 'random':
                    chosen = rng.sample(remaining, k)
                elif strategy == 'left_to_right':
                    chosen = remaining[:k]
                else:
                    chosen = remaining[-k:][::-1]
                selected = set(chosen)
                remaining = [p for p in remaining if p not in selected]
                positions = torch.tensor(chosen, device=device, dtype=torch.long)
                local = positions - lo
            tokens = predictions[local]
            x[0, start + positions] = tokens
            revealed[positions] = True
            trace.append(dict(block=block, positions=positions.clone(), token_ids=tokens.clone(),
                              confidences=confidence[local].clone(), forward_passes=1))
            del logits, predictions, confidence
    if gpu:
        end.record()
        end.synchronize()
        gpu_seconds = begin.elapsed_time(end) / 1000
    else:
        gpu_seconds = None
    wall_seconds = time.perf_counter() - wall_start
    def serialize(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serialize(item) for item in value]
        return value
    trajectory = [dict(step=i, **serialize(record)) for i, record in enumerate(trace)]
    return x[0, start:].cpu().tolist(), gpu_seconds, wall_seconds, trajectory
