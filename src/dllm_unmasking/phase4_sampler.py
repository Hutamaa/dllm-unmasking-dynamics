"""32-pass spaced decoding with at most one revision per previously committed token."""
import random
import time
import torch
from dllm_unmasking.phase3_schedule import schedule
from dllm_unmasking.phase2_samplers import predictions,serialize

STRATEGIES=('random_remask','low_confidence_remask')


@torch.inference_mode()
def generate(model,prompt,strategy,*,mask_id=126336,revision_seed=314159):
    if strategy not in STRATEGIES or prompt.ndim!=2 or prompt.shape[0]!=1:
        raise ValueError('Known remasking policy and one unpadded prompt required')
    device=prompt.device;start=prompt.shape[1]
    x=torch.full((1,start+256),mask_id,dtype=torch.long,device=device);x[:,:start]=prompt
    committed=set();revised=set();stored_confidence=torch.zeros(256,device=device)
    rng=random.Random(revision_seed) if strategy=='random_remask' else None
    trace=[];gpu=device.type=='cuda'
    if gpu:
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
    wall_start=time.perf_counter()
    if gpu:begin.record()
    for step,new in enumerate(schedule()):
        masks_before=int((x[0,start:]==mask_id).sum().item())
        assert masks_before==256-8*step
        assert len(new)==8 and not committed.intersection(new)
        eligible=sorted(committed-revised)
        budget=min(2,len(eligible)) if step else 0
        if strategy=='random_remask':
            selected=rng.sample(eligible,budget)
        else:
            candidates=torch.tensor(eligible,dtype=torch.long,device=device)
            rank=torch.argsort(stored_confidence[candidates],stable=True)[:budget]
            selected=candidates[rank].cpu().tolist()
        assert len(set(selected))==budget and set(selected).issubset(committed-revised)
        rev_positions=torch.tensor(selected,dtype=torch.long,device=device)
        original_tokens=x[0,start+rev_positions].clone()
        original_confidence=stored_confidence[rev_positions].clone()
        # Remask before the single forward. Prompt and not-yet-committed positions are never revision candidates.
        x[0,start+rev_positions]=mask_id
        assert int((x[0,start:]==mask_id).sum().item())==masks_before+budget
        tokens,confidence,_=predictions(model,x,start,mask_id)
        all_positions=torch.tensor(new+selected,dtype=torch.long,device=device)
        if not bool(torch.isfinite(confidence[all_positions]).all().item()):
            raise RuntimeError('Nonfinite selected-token probabilities')
        if not bool(((confidence[all_positions]>=0)&(confidence[all_positions]<=1)).all().item()):
            raise RuntimeError('Selected probabilities outside [0,1]')
        # One commitment for both new tokens and freshly predicted revisions, from the same masked state.
        x[0,start+all_positions]=tokens[all_positions]
        stored_confidence[all_positions]=confidence[all_positions]
        committed.update(new);revised.update(selected)
        masks_after=int((x[0,start:]==mask_id).sum().item())
        assert masks_after==masks_before-8
        revision_records=[dict(position=p,original_token=original_tokens[j],replacement_token=tokens[p].clone(),
            original_confidence=original_confidence[j],replacement_confidence=confidence[p].clone(),
            revision_step=step,decoding_half='first' if step<16 else 'second',
            changed=original_tokens[j]!=tokens[p]) for j,p in enumerate(selected)]
        trace.append(dict(step=step,positions=new,token_ids=tokens[new].clone(),confidences=confidence[new].clone(),
            revisions=revision_records,eligible_revision_count=len(eligible),revision_budget=budget,
            forward_passes=1,remaining_masks_before=masks_before,remaining_masks_after=masks_after))
    assert len(committed)==256 and len(revised)==62 and int((x[0,start:]==mask_id).sum().item())==0
    if gpu:end.record();end.synchronize();seconds=begin.elapsed_time(end)/1000
    else:seconds=None
    return x[0,start:].cpu().tolist(),seconds,time.perf_counter()-wall_start,serialize(trace)
