# X/Twitter thread draft (post from anon account; PDF attached to post 1)

**1/**
I spent a month doing surgery on a 1-bit 27B model. No training, no cloud, one 64GB laptop.

Deleted 15% of its bytes: GSM8K didn't move. Then repaired the folded model by flipping 1% of its sign bits with a single linear solve.

Paper attached. Every claim benchmarked. 🧵

**2/**
The format ships a plane of pure redundancy: every stored bias equals f16(−s/2), verified on all 200M+ groups.

Delete it: −420 MB on disk, logits unchanged. Re-derive it inside the kernels: −420,225,024 bytes of RAM, measured. Free.

**3/**
The depth map (all 64 blocks screened): first and last blocks are sacred, slack lives mid-depth, and damage composes additively (~1.1x tax).

Best composition: 4 blocks + 3 deep attention sublayers = −709 MB (−15.4%), macro .8413 → .7963, GSM8K .910 → .910.

What suffers is breadth (MMLU −8), not reasoning.

**4/**
The metadata: after bias derivation the model is a 3.77 GB sign plane (incompressible, 8.000 bits/byte measured) + 420 MB of f16 scales.

8-bit scales: free. 4-bit: −2.9 macro. Sharp knee.

Combined artifact: 3.82 GB (−18.9%) at .7900.

**5/**
The weird one: you can make a folded 1-bit model BETTER by editing bits.

Fit the dropped blocks' "linear shadow" (one ridge solve, 64 calibration sequences), absorb into 3 surviving projections, requantize into the format. 1% of signs flip, zero bytes added.

+1.0 macro at identical bytes, over 500 items.

**6/**
The real thesis: SIX times, cheap proxy metrics lied and only the benchmark told the truth.

Worst case: an evolutionary search over 311 architectures that dominated the KL frontier at every byte tier, and whose champion benched BELOW the hand-built config it "beat." It won the screen game; the bench exposed the game.

**7/**
The pattern: deleting redundant structure is cheap per screen-unit. Densely perturbing every token's computation (metadata noise, head removal) is catastrophic per screen-unit. Teacher-forced KL can't see dense perturbations compounding over 2,000-token generations.

If your pruning eval is perplexity or KL: benchmark or it didn't happen.

**8/**
Generality (dense 8B sibling): every structural law replicates, but the same 4-block recipe that costs 2.4 pts on the 27B costs 14.8 there. Foldability is a property of over-provisioned depth.

All of it on one MacBook. PDF attached. Code on publication.

..............................
## Posting notes (not part of thread)
- Attach paper.pdf to post 1 (or host on an anon GitHub release and link).
- Optional images: Table 2 (the ladder) for post 3; Table 5 (six divergences) for post 6.
- Nothing here identifies you, your machine, or the company. Do not link the private repo.
