"""Fixed two-pass interleave inside successive 16-position tiles."""


def schedule():
    # t=2*b+r; position=16*b+r+2*j (b=0..15, r=0,1, j=0..7).
    batches=[[16*b+r+2*j for j in range(8)] for b in range(16) for r in (0,1)]
    flat=[p for batch in batches for p in batch]
    if len(batches)!=32 or sorted(flat)!=list(range(256)) or any(len(set(batch))!=8 for batch in batches):
        raise RuntimeError('Invalid spaced schedule')
    if any(min(y-x for x,y in zip(batch,batch[1:]))!=2 for batch in batches):
        raise RuntimeError('Invalid spacing')
    return batches
