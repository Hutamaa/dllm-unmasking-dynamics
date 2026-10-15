"""Phase 2 parallel commitment. Phase 1 samplers are never imported or modified."""
import random
import time
import torch

STRATEGIES = ('random', 'left_to_right', 'right_to_left', 'confidence', 'batched_influence')


def validate_schedule(gen_length, block_length, steps):
    if (gen_length, block_length, steps) not in ((256,256,64),(256,256,32)):
        raise ValueError('Phase 2 permits only length/block 256 with 64 or 32 steps; no sequential rerun')
    return gen_length // steps


def expected_passes(steps, strategy):
    k = validate_schedule(256,256,steps)
    return steps + sum(min(16,m) for m in range(256,0,-k)) if strategy == 'batched_influence' else steps


def predictions(model, x, start, mask_id, entropy=False):
    output = model(x, use_cache=False)
    logits = output.logits[0,start:].float().clone()
    del output
    logits[:,mask_id] = -torch.inf
    tokens = logits.argmax(dim=-1)
    log_probs = logits - torch.logsumexp(logits,dim=-1,keepdim=True)
    confidence = log_probs.gather(1,tokens[:,None]).squeeze(1).exp()
    if not entropy:
        return tokens,confidence,None
    probs = log_probs.exp()
    log_probs[:,mask_id] = 0
    return tokens,confidence,-(probs*log_probs).sum(dim=-1)


def influence_candidates(model, x, start, revealed, tokens, confidence, entropy, mask_id, k):
    """Independent serial probes; all winners use baseline tokens, never probe tokens."""
    eligible = ~revealed
    rank = confidence.masked_fill(~eligible,-torch.inf)
    count = min(16,int(eligible.sum().item()))
    candidates = torch.argsort(rank,descending=True,stable=True)[:count].sort().values
    influence = torch.full_like(confidence,-torch.inf)
    for position in candidates.cpu().tolist():
        probe = x.clone()
        probe[0,start+position] = tokens[position]
        _,_,after = predictions(model,probe,start,mask_id,entropy=True)
        other = eligible.clone()
        other[position] = False
        influence[position] = (entropy[other]-after[other]).sum()
    # Scores start in position order; stable ties favor the lower position.
    selected = torch.argsort(influence,descending=True,stable=True)[:k]
    return selected,influence[selected],dict(candidate_positions=candidates,
        candidate_scores=influence[candidates],candidate_confidences=confidence[candidates]),count


def serialize(value):
    if isinstance(value,torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value,dict):
        return {k:serialize(v) for k,v in value.items()}
    if isinstance(value,list):
        return [serialize(v) for v in value]
    return value


@torch.inference_mode()
def generate(model,prompt,strategy,*,steps,gen_length=256,block_length=256,mask_id=126336,seed=42):
    k = validate_schedule(gen_length,block_length,steps)
    if strategy not in STRATEGIES or prompt.ndim!=2 or prompt.shape[0]!=1:
        raise ValueError('Unknown strategy or expected one unpadded prompt')
    start=prompt.shape[1];device=prompt.device
    x=torch.full((1,start+256),mask_id,dtype=torch.long,device=device)
    x[:,:start]=prompt
    revealed=torch.zeros(256,dtype=torch.bool,device=device)
    remaining=list(range(256));rng=random.Random(seed);trace=[]
    gpu=device.type=='cuda'
    if gpu:
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
    wall_start=time.perf_counter()
    if gpu:begin.record()
    for step in range(steps):
        tokens,confidence,entropy=predictions(model,x,start,mask_id,entropy=strategy=='batched_influence')
        passes=1;extra={}
        if strategy=='batched_influence':
            positions,scores,candidates,count=influence_candidates(model,x,start,revealed,tokens,confidence,entropy,mask_id,k)
            passes+=count
            extra=dict(influence_scores=scores,candidate_pool=candidates,influence_top_k=16,
                       influence_policy='common_state_top_k_simultaneous')
        elif strategy=='confidence':
            rank=confidence.masked_fill(revealed,-torch.inf)
            positions=torch.argsort(rank,descending=True,stable=True)[:k]
        else:
            if strategy=='random':chosen=rng.sample(remaining,k)
            elif strategy=='left_to_right':chosen=remaining[:k]
            else:chosen=remaining[-k:][::-1]
            chosen_set=set(chosen);remaining=[p for p in remaining if p not in chosen_set]
            positions=torch.tensor(chosen,dtype=torch.long,device=device)
        # The only permanent write in the step. Every selected token is a baseline prediction.
        selected_tokens=tokens[positions]
        x[0,start+positions]=selected_tokens
        revealed[positions]=True
        trace.append(dict(step=step,positions=positions.clone(),token_ids=selected_tokens.clone(),
                          confidences=confidence[positions].clone(),forward_passes=passes,**extra))
    if gpu:
        end.record();end.synchronize();seconds=begin.elapsed_time(end)/1000
    else:seconds=None
    wall=time.perf_counter()-wall_start
    trajectory=serialize(trace)
    positions=[p for t in trajectory for p in t['positions']]
    if len(positions)!=256 or len(set(positions))!=256 or any(len(t['positions'])!=k for t in trajectory):
        raise RuntimeError('Reveal schedule failed: expected 256 distinct positions')
    return x[0,start:].cpu().tolist(),seconds,wall,trajectory
