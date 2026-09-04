# X/Twitter thread draft (post from anon account; PDF attached to tweet 1 or linked)

**1/**
I spent a month doing surgery on a 1-bit 27B model with no training, no cloud — just one 64GB laptop.

Deleted 15% of its bytes. GSM8K didn't move. Then I repaired it by flipping 1% of its sign bits with a single linear solve.

Paper (PDF below). Everything benchmarked. 🧵

**2/**
Setup: a public 27B quantized to 1.125 bits/weight (weights are literally ±s/2, one f16 scale per 128 weights). 4.71 GB.

Rules: no training, no calibration-based weight updates, and surviving weights must stay BYTE-IDENTICAL. Every claim gated by a 500-item benchmark.

**3/**
First finding: the format ships a plane of pure redundancy.

Every stored bias satisfies b = f16(−s/2). Exactly. All 200M+ groups.

Delete it: −420 MB on disk, logits bit-identical. Re-derive it in-register in the kernels: −420,225,024 bytes of RAM, measured, bit-exact.

**4/**
Then the depth map. Screening all 64 blocks: first & last blocks are sacred, slack lives mid-depth, and damage composes ADDITIVELY (tax ~1.1x).

Best composition: drop 4 blocks + 3 deep attention sublayers = −709 MB (−15.4%).

Macro: .8413 → .7963. GSM8K: .910 → .910. Untouched.

**5/**
That's the headline artifact: 4.2 GB, byte-identical survivors, math reasoning at full reference parity.

What gets damaged instead? Breadth. MMLU takes −8. The reasoning spine survives folding; diffuse knowledge is what you're deleting.

**6/**
Next: the scales metadata. After bias-derivation the model is a 3.77 GB sign plane (measured 8.000 bits/byte — incompressible) + 420 MB of f16 scales.

8-bit scales: FREE (−0.5 macro, within noise). −192 MB.
4-bit scales: −2.9 macro. The knee is sharp.

Combined artifact: 3.82 GB (−18.9%) at .7900.

**7/**
Now the weird one. Can you make a folded 1-bit model BETTER by editing bits?

Yes. Fit the "linear shadow" of the dropped blocks (one ridge regression on 64 calibration sequences), absorb it into 3 surviving projections, requantize back into the format.

1% of sign bits flip. Zero bytes added.

**8/**
Result: +1.0 macro at identical bytes (.7963 → .8063, benchmarked over 500 items).

Training-free, closed-form, in-format repair of a 1-bit model. The sign/scale lattice around a folded model contains strictly better points, one linear solve away.

**9/**
(Its limits, mapped honestly: doesn't compose with scale quantization — bench got WORSE despite an identical screen profile. Doesn't reach deeper folds. Works for moderate structural damage, one perturbation at a time.)

**10/**
The real thesis of the paper though: SIX times, cheap proxy metrics lied and only the benchmark told the truth.

Worst offender: an evolutionary search over 311 architectures that provably dominated the KL frontier at every byte tier… and benched BELOW the hand-built config it "beat."

It won the screen game. The bench exposed the game.

**11/**
And a config of 8 "redundant" GQA head-groups that screened 6x cheaper than the gentlest block drop — and benched 6x worse. −3.5 points for 30 MB.

Pattern across all six: DELETING redundant structure is cheap per screen-unit. DENSELY PERTURBING every token's computation (metadata noise, head removal) is catastrophically expensive per screen-unit.

**12/**
Teacher-forced KL scores one forced step at a time. It cannot see how dense small perturbations compound over a 2,000-token generation.

If you prune/quantize/edit models and your eval is perplexity or KL: your numbers are only comparable within one operator family. Benchmark or it didn't happen.

**13/**
Generality check on a dense 8B sibling: every structural law replicates (boundary protection, mid-depth slack, additive composition). But the SLACK doesn't — the same 4-block recipe that costs 2.4 pts on the 27B costs 14.8 there.

Foldability is a property of over-provisioned depth, not of 1-bit models.

**14/**
Everything — 20+ full benchmark runs, kernel work, the search, the repairs — on one 64 GB MacBook. No cluster. The constraint was the method: when compute is scarce you build ladders, and when you build ladders you learn which rungs lie.

PDF: [attach/link]
Code on publication.

---
## Posting notes (not part of thread)
- Attach paper.pdf to tweet 1 (or upload to an anon GitHub release / file host and link).
- Optional images: screenshot Table 1 from the PDF for tweet 4/5; the six-divergences list for tweet 10.
- Nothing in the thread or PDF identifies you, your machine username, or the company. Do not link the private repo.
