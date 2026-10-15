"""Only ltr_spaced: one baseline forward and one simultaneous commit per step."""
import time
import torch
from dllm_unmasking.phase3_schedule import schedule
from dllm_unmasking.phase2_samplers import predictions,serialize


@torch.inference_mode()
def generate(model,prompt,*,mask_id=126336):
    if prompt.ndim!=2 or prompt.shape[0]!=1:raise ValueError('One unpadded prompt required')
    start=prompt.shape[1];device=prompt.device
    x=torch.full((1,start+256),mask_id,dtype=torch.long,device=device);x[:,:start]=prompt
    trace=[];gpu=device.type=='cuda'
    if gpu:
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
    wall_start=time.perf_counter()
    if gpu:begin.record()
    for step,batch in enumerate(schedule()):
        tokens,confidence,_=predictions(model,x,start,mask_id)
        positions=torch.tensor(batch,dtype=torch.long,device=device)
        chosen=tokens[positions]
        # Only permanent update in this step; every token uses the same baseline.
        x[0,start+positions]=chosen
        trace.append(dict(step=step,positions=positions,token_ids=chosen.clone(),
                          confidences=confidence[positions].clone(),forward_passes=1))
    if gpu:end.record();end.synchronize();seconds=begin.elapsed_time(end)/1000
    else:seconds=None
    return x[0,start:].cpu().tolist(),seconds,time.perf_counter()-wall_start,serialize(trace)
