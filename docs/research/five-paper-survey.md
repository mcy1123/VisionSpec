# Five-paper survey for GLANCE (arXiv 2609.00355) / Draft-OPD multimodal

Scope: KnapSpec, SelfJudge, HIPPO, Vegas, FLy. All five read from the PDFs (method sections, algorithm
boxes, results tables). Every number below was read in the PDF; "not stated" marks absent numbers.
None of the five papers evaluates a block-diffusion tree drafter, a vision-language target, or an
on-policy-distilled block head, so **no number below is transferable evidence** — only mechanisms are.

System under improvement (for reference): frozen Qwen3-VL-8B target; 1.05B 5-layer block-diffusion head
reading target fused states at kept layers {1, 9, 17, 25, 33}, block B=16 filled in one pass; prefix-closed
tree of budget N (63 nodes, branch factor 8) verified in ONE ancestor-masked target pass; commit the
longest target-greedy path; lossless/bitwise-identical. ChartQA tau 5.16 / 3.42x.

---

## 1. KnapSpec — Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem

**Citation.** Seongjin Cha, Gyuwan Kim, Dongsu Han, Tao Yang, Insu Han. ICML 2026 (PMLR 306); arXiv:2602.20217v2.
Training-free self-speculative decoding that reformulates *which layers to keep in the draft sub-network* as a
0/1 knapsack maximizing tokens-per-wall-clock, with a formal cosine-similarity guarantee.

**Genuinely new component.**
1. **TPT (Tokens-per-Time) objective**, Eq. (4) — the quantity every prior SSD proxy (TPL, cosine sim) only
   approximates. Measured verbatim from the PDF:
   `TPT(S, γ) = [(1 − α_S^{γ+1}) / (1 − α_S)] · [1 / (γ · t_Draft(S) + t_Target)]`
   with `t_Draft(S) = n_Attn(S)·t_Attn + n_MLP(S)·t_MLP` and `t_Target = L(t_Attn + t_MLP)`. Latency
   coefficients are fit once by hardware profiling; attention is Θ(n) in context length, MLP is Θ(1).
2. **0/1 knapsack formulation**, Eq. (6): items are individual Attention and MLP modules, weight = latency,
   value = cosine similarity to the target's own hidden states:
   `argmax_{S⊆[2L]} α_S ≈ argmax_{S⊆[2L]} cos(f(X), f^{(S)}(X))  s.t.  n_Attn(S)w_Attn + n_MLP(S)w_MLP = k`
   solved by a **single parallel DP pass over all budgets at once** (Algorithm 1: `g[i,j]` keeps the
   hidden state with the better cosine at layer `i`, skip-budget `j`, with pruning at `τ = 0.5` and
   skip-weight > K/2), then a grid search over (candidate set A) × (draft length γ ≤ D=10).
3. **Lemma 4.1** — the first proof that cosine similarity *suffices* for greedy agreement: for target
   embedding x with top-1 margin `ξ(x) = ⟨w_{i*},x⟩ − max_{j≠i*}⟨w_j,x⟩`, if
   `cos(x, x') ≥ 1 − ξ(x)² / (2‖x‖²₂ max_{j≠i*}‖w_{i*} − w_j‖²₂)` then `argmax` agrees. The paper's own
   caveat: the derivation assumes `‖x'‖₂ = ‖x‖₂`, "empirically supported by RMSNorm".

**What it needs.** No training, no extra modules, no auxiliary model. Needs (i) a one-time hardware profiling
step for `t_Attn(n)`, `t_MLP`; (ii) hidden-state access at every layer of the target (it is the target's own
sub-network); (iii) a re-optimization interval (T=64 steps for <10B, T=128 above) with reported overhead
O(nL) time / O(L) memory. Assumes a Transformer target where attention and MLP can be independently skipped —
i.e. a *self*-speculative draft. It does **not** assume a tree, a multimodal target, or a block drafter.

**Applicability to GLANCE.** Partially direct, and it maps to two named open problems.
- **Tree budget allocation — direct.** GLANCE fixes N=63 and states "The budget is a small fixed constant,
  and Section 5.4 shows returns flatten once the width is moderate" and "Growing N adds verifier FLOPs inside
  that single pass but no sequential draft steps". That is exactly the saturation TPT predicts, and GLANCE has
  the explicit cost model TPT needs (verifier FLOPs grow with N; drafting cost is N-independent). KnapSpec
  supplies the objective (maximize tokens-per-time, not accepted length) and an algorithm to solve it.
- **Verification cost — direct.** KnapSpec's own §5.1 finding on its Table 2: methods with *higher* acceptance
  rate still lose on throughput; TPT correlates 0.837 with throughput vs 0.538 for acceptance rate.
- **Drafter plateau — indirect.** GLANCE's head reads five *fixed* kept layers {1, 9, 17, 25, 33}. KnapSpec's
  knapsack is literally a layer-subset selector with a latency-aware value function, so "which layers to fuse,
  and how many, as a function of context length" is a re-derivation away. But the head only *reads* states the
  target computes anyway, so the latency saving is the head's cross-attention cost, not target-layer cost —
  the payoff is much smaller here than in KnapSpec's own setting.
- **Does not address:** high-entropy free text, state-distribution mismatch, drafter plateau via training.

**Expected effect size, with regime caveats.** KnapSpec reports **1.06×–1.47× wall-clock** across 8 model/task
cells in Table 2 (best: 1.47× on GovReport with Llama3.1-70B, vs SWIFT 1.33× and CLaSp 1.22×), and up to 1.23×
under nucleus sampling in Table 3, over 1B–70B **text** models. **Not transferable**: those numbers are
layer-skip SSD on long-context LLM text, no tree, no vision, no block drafter, and KnapSpec's own acceptance rates in
Table 2 are 84.0–97.0% — a regime GLANCE never reaches on free text. Transferable claim is only that
throughput-optimal budget ≪ length-optimal budget, worth a few percent to low-double-digit percent at best
given GLANCE's own saturation report. **Not stated:** any VLM, tree, or block-diffusion result.

**Cheapest falsifying experiment (<1 day, 2 GPUs, inference only).**
Sweep the tree budget N ∈ {15, 31, 63, 127, 255} on ChartQA and COCO-captioning with the existing engine,
logging per round: τ, draft-pass time, verify-pass time, tokens/sec. Fit `TPT(N) = τ(N)/(t_draft + t_ver(N))`
and compare `argmax_N TPT` with the shipped N=63. **Falsified if TPT is flat or still rising at N=63 on all
five tasks** — then GLANCE's fixed budget is already throughput-optimal and the idea has no room. Second,
cheap sub-check of Lemma 4.1: compute `ξ(x)` over the Qwen3-VL-8B vocabulary and evaluate the bound; if the
required cosine is ≈0.999+ the certificate is vacuous for this target (vocabulary-margin-dependent, so this is
a real possibility).

**Confidence: medium.** The objective is right and GLANCE's own §5.4 is the opening, and the experiment is
pure inference so a negative result costs a day. But the mechanism is a re-derivation rather than a drop-in
module, the head's layer/fusion cost is a small share of round latency, and GLANCE already reports saturation —
so the likely effect is small.

---

## 2. SelfJudge — Faster Speculative Decoding via Self-Supervised Judge Verification

**Citation.** Kanghoon Yoon, Minsub Kim, Sungjae Lee, Joonhyung Lee, Sunghyeon Woo, Yeonjun In, Se Jung Kwon,
Chanyoung Park, Dongsoo Lee. ICML 2026 (PMLR 306); arXiv:2510.02329v2. Self-supervised training-data
construction for a relaxed ("judge") verifier that accepts semantically-preserving draft tokens.

**Genuinely new component.** The **semantic preservation score** and the labeling rule built on it, Eq. (6)/(7):
`s(y, z_i) = log P(z_i | y_{<i}, y_{>i}) − log P(y_i | y_{<i}, y_{>i})`, decomposed as
`s(y, z_i) = s_prefix(y, z_i) + log[P(y_{>i} | z_i, y_{<i}) / P(y_{>i} | y_i, y_{<i})]` (a Log Bayes Factor).
Label a substituted (draft) token **acceptable iff s > τ**, else unacceptable; train a logistic-regression
verifier on the target's own hidden states (`Verifier(h_t) > θ ⇒ accept`), applied at inference *in parallel
with* rejection sampling: a token is accepted if either approves. Labels are only used offline; the verifier is
forward-only at inference. Suffix length N=20 was best; **N=0 (prefix-only) was worst** — the bidirectional
term is the contribution. Threshold τ is set to `quantile(scores of AutoJudge-unacceptable tokens, 0.1)`.

**What it needs.** Offline target forwards on token-substituted sequences `[y_{<i}, z_i, y_{>i}]` (prefill only,
no completion). 69,432 labels from 1,220 prompts; 53,318 labels for the 70B variant, **8.5 h on 4×H100**
(Table 8) — vs 120 h for AutoJudge, 14× cheaper. One extra linear classifier (2,000 params scale, 0.02 s vs
7.515 s total verification time, 0.26%). Assumes the target can be run on teacher-forced substituted
sequences and that a linear probe on its hidden states can separate accept/reject.

**Applicability to GLANCE.** **Not applicable as a verification method, because GLANCE is lossless and
bitwise-identical to greedy by design** (Theorem 1) — SelfJudge's entire value is buying speed by accepting
tokens the target would not have emitted. It is also a *chain* verifier (accept/reject per token), not a tree.
The transferable piece is the **offline label**, aimed at the Draft-OPD open problem *"why does reverse KL on
the first rejected position help, and why does OPD lose on long free text"*: on high-entropy free text, a large
share of "rejected" anchors are semantically fine (`"It is" → "It's"`, `", → ."`, `"solve → find"` per §4.7), so
alpha_rev punishes the drafter for a token the target itself considers meaning-preserving. Concretely:
soft-weight or gate the reverse-KL term at the first rejected position by `s(y, z_i)` instead of a hard 0/1
reject. It does **not** address tree budget allocation or verification cost.

**Expected effect size, with regime caveats.** Table 1 (Llama-3.1-8B / Llama-3.2-1B): SelfJudge-F moves
average accepted length from SD's 9.14 → 11.29 on GSM8K and 4.36 → 6.38 on MMLU, average Δm **+2.17** at
Δtask **−1.6%** (SelfJudge-R: Δm +0.81, Δtask −0.5%). Wall-clock Table 3: 111.21 → 137.37 tok/s on GSM8K
(+23.5%), 61.13 → 89.45 on MMLU (+46.3%). Table 2 on 70B/8B: Δm +2.09 to +2.29 with no accuracy loss.
**Not transferable**: text-only, 1B autoregressive drafter, γ=15–20, batch-1 A100, and the mechanism is
deliberately *distribution-relaxing*, which is the opposite of GLANCE's contract. (Abstract's "+2.06 accepted
tokens with only −1.0%" disagrees slightly with Table 1's +2.17/−1.6; the table is what I trust.)

**Cheapest falsifying experiment (<1 day, 2 GPUs).**
From existing Draft-OPD tree-rollout logs (which already record `anchor = (pos, proposed tokens, n_accepted)`),
take ~2–5k first-rejected anchors and compute `s(y, z_i)` with N=20 on the frozen target (prefill only;
minutes, not the 8.5 h full-label job). Split anchors into high-s / low-s. Replay the drafter twice with
alpha_rev applied only to the low-s split vs to all rejections, 3 seeds each. **Falsified if the split does not
beat constant alpha_rev=0.5** — i.e. if the semantic-preservation score does not predict where reverse KL
actually helps. Secondary read-out is free: the fraction of high-s rejections on free text vs ChartQA directly
tests the "OPD loses on long free text because its rejections are semantically fine" hypothesis.

**Confidence: medium-low as a verification method (it contradicts losslessness), medium as a training-signal
relabeling for Draft-OPD.** The label is well-defined, cheap, and reimplementable; the risk is that it explains
rather than improves.

---

## 3. HIPPO — Holistic-Aware Parallel Speculative Decoding for Video-LLMs

**Citation.** (authors not captured in the extracted text header) arXiv:2601.08273. Video-LLM acceleration via
target-guided visual-token pruning for the draft model plus overlapped draft/verify scheduling.

**Genuinely new component.** Two separable parts.
1. **Semantic-aware token preservation** for the *draft* model. Three scores are fused, Eq. (2)–(7):
   `s_attn(v_j)` = language-to-video cross-attention averaged over text tokens, layers, heads (computed from
   the prefilled KV cache by KV reconstruction to avoid the full O(L²) matrix);
   `s_temp(v_j^f) = 1 − (1/|N(f)|) Σ_{f'∈N(f)} cos(v_j^f, v_j^{f'})` (inverted cross-frame similarity — static
   background is redundant); `s_spa(v_j^{f,c})` = variance of intra-crop pairwise cosine similarity
   (`S_c = E_c E_cᵀ`) as an information-content proxy. Each score is z-normalized per frame and summed:
   `s(v_j) = ŝ_attn(v_j) + ŝ_spa(v_j) + ŝ_temp(v_j)`; keep top 10% of visual tokens for the draft.
2. **Video Parallel Speculative Decoding (VPSD).** Synchronous draft-target prefill (the draft decodes while the
   target finishes its vision-encoder prefill), then adaptive switching: after a fully-accepted batch, the draft
   speculatively generates batch t+1 *concurrently with* the target verifying batch t ("optimistic"); after any
   rejection, the target verifies x₁ while the draft generates x₂..x_γ ("conservative"), aborting on x₁
   rejection.

**What it needs.** Two *full VLM models* (7B draft, 32B/72B target), each with its own visual encoder; access to
the target's cross-attention maps and the draft's KV cache; a serving engine that can run draft and target
concurrently. Motivation numbers: draft time is 30.7–38.3% of round time across four backbones (their Fig. 4).

**Applicability to GLANCE.** **The token-pruning half is not applicable, because GLANCE's drafter never reads
raw visual tokens** — its cross-attention KV come from the target's fused hidden states, so "give the drafter
fewer visual tokens" has no referent; there is no separate visual encoder to starve. The finding itself
(position bias: attention-based selection picks >44%, peaking 57.8%, boundary tokens when boundaries are only
~20% of tokens; random 10% beats attention-only on MLVU) is a real VLM result but irrelevant to a head that
inherits grounding for free. Only **VPSD** transfers, to the open problem *verification cost / draft latency*:
overlap round t+1's block-head pass with round t's verification. This is engineering, not a method, and its
ceiling is the head's share of round latency (a 1.05B head on 5 fused layers vs an 8B full forward — small,
very unlike HIPPO's 7B-vs-72B pair). It also cannot be done naively: the block head conditions on the target's
fused states at the *accepted* prefix, which is unknown until verification finishes, so optimistic drafting
means drafting from a possibly-wrong root — exactly HIPPO's bet, but with GLANCE's lossless commit rule
requiring the discarded speculative pass to be free.

**Expected effect size, with regime caveats.** Up to **3.51×** speedup, peak MAT 12.42 (MLVU,
video-SALMONN2+). The closest row to GLANCE's regime is Qwen3-VL 7&32B on VideoMME: vanilla SD 3.39 MAT /
1.35×, SD-tree 3.91 / 1.73×, SpecVLM 3.87 / 1.89×, HIPPO 4.67 / 2.38× — i.e. +38% MAT over vanilla SD, and
HIPPO still only *matches* GLANCE's reported ChartQA τ of 5.16 on a different benchmark. **Not transferable**:
video (long temporal context) not single-image; a 7B full VLM drafter with its own encoder, not a 1.05B
state-reading head; no tree; the speedup is graded by a cost ratio ~10× more draft-heavy than GLANCE's.
**Not stated:** any text-LLM or non-video result, any tree-budget result.

**Cheapest falsifying experiment (<1 day, 1 GPU, no training).**
Instrument the GLANCE serving loop and record `t_head / (t_head + t_verify)` per round on ChartQA and COCO.
**Falsified if the head's share is below ~10%** — then overlapping it with verification cannot buy more than
~10% and VPSD is not worth a paper for this system. (Secondary, <1 day: instrument the *optimistic* variant —
draft block t+1 from the pre-verification root — and measure how often the root survives; that hit rate
multiplies the whole benefit.)

**Confidence: low.** The transferable component is a scheduling change whose upside is bounded by a cost ratio
that is ~10× less favourable here than in HIPPO's setting, and HIPPO's headline evidence comes from a mechanism
(visual-token pruning) that GLANCE structurally cannot use.

---

## 4. Vegas — Self-Speculative Decoding with Verification-Guided Sparse Attention

**Citation.** Yikang Yue, Yuqi Xue, Jian Huang. arXiv:2602.07223v2. Self-speculative decoding where the
verification pass *doubles as* the KV-selection oracle for the next drafting pass.

**Genuinely new component.** Verification-guided KV selection. During verification the engine already computes
full attention over the prefix for every draft-token query; Vegas intercepts those logits and keeps the k prefix
tokens that maximize **cumulative** attention-logit coverage over the whole draft chain, Eq. (2):
`T = argmax_{T⊆P, |T|=k} Σ_{t=1}^{γ+1} Σ_{h∈H} Σ_{i∈T} l^{(h)}_{ti}`, and the deployable two-query
approximation, Eq. (3): `T = argmax_{T⊆P, |T|=k} Σ_{h∈H} Σ_{i∈T} (l^{(h)}_{1i} + l^{(h)}_{(γ+1)i})`
first draft token + bonus token, then average over heads and take top-k. The second, sharper finding: **include
the discarded/rejected draft tokens in the coverage.** Their justification: "many draft tokens possessing a high
acceptance probability are rejected due to probabilistic sampling … these rejected tokens often remain
semantically aligned with the target generation", so their attention weights also estimate KV criticality. Their
ablation: dropping discarded tokens costs **3%–14%** of accepted tokens per iteration at γ=7 (Qwen3-8B and
gpt-oss-20b). Implementation: instrumented FlashAttention-3 writing BF16 logits to HBM post-mask/pre-softmax,
and a page-size-1 PagedAttention repurposing to make token-granular selection free.

**What it needs.** A self-speculative drafter that has *its own KV cache over the same prefix* and can run
sparse attention over a chosen subset. Kernel-level modifications (FA3 + vLLM). Two-query logit collection
overhead 5.9%–9.4%; hyperparameters (sparsity ratio, γ) tuned per model in three steps.

**Applicability to GLANCE.** **The KV-selection method does not transfer, because GLANCE's drafter has no
separate KV cache to sparsify** — the block head cross-attends to the target's fused hidden states in one pass
over a 16-token block, not to a prefix KV cache it maintains. What *does* transfer is the accepted-tokens
finding, and it is the single most useful piece of evidence across all five papers for the Draft-OPD open
problem *"why does tree rollout training work"*: Vegas independently reports that the *rejected* branches carry
real information about the target's plausible continuation, which is a mechanistic account of why training on
tree rollout (where rejected candidates come from the drafter's own state distribution) beats chain rollout,
and of why a reverse-KL term on the first rejected position helps at all. It also yields a concrete variant:
weight alpha_rev by the rejected token's own target-side acceptance probability rather than applying a flat
0.5 (high-probability rejections ⇒ the drafter's state distribution was already right; punish them less).
It does **not** address tree budget allocation or drafter plateau.

**Expected effect size, with regime caveats.** 1.25×–2.81× throughput over default vLLM and **1.15×–1.29×**
over state-of-the-art sparse self-speculative decoding; 18%–29% higher throughput than the best baseline on
96K–120K contexts; KV-selection overhead 5.9%–9.4% vs 21.7% (MagicDec-Quest) and 11.2%–29.1% (SpecExtend).
Table 1 (LongBench-v2, 7% KV): All-Draft-Tokens 6.13 vs 1st-Draft+Bonus 6.11 accepted tokens at γ=7 on
Qwen3-8B (6.25 vs 6.24 on gpt-oss-20b). **Not transferable at all**: text LLMs, 96K–128K contexts, 7% KV
sparsity, batch 4–128 on H100 NVL, single-token chain drafts — no tree, no vision, no block head, and the
speedup is against vLLM's *sparse-attention-free* serving configuration. **Not stated:** any VLM or tree result.

**Cheapest falsifying experiment (<1 day, 2 GPUs).**
Two independent halves, either can kill it. (a) *Insight test, ~hours, 1 GPU:* in the existing tree rollout,
for every first-rejected node record the target's own probability of the rejected draft token (already in the
verification pass's row — no extra forward). Bucket anchors by that probability and re-run the replay with
alpha_rev scaled by it vs a flat alpha_rev=0.5, 3 seeds. **Falsified if the probability-weighted alpha_rev does
not beat the flat one** — then "rejected tokens are informative" is not actionable for the training signal.
(b) *Direct method test, ~1 day:* because GLANCE's head has no cache, the closest analog would be sparsifying
the *head's* cross-attention over the 5 fused layers; check first whether that attention is even diffuse enough
to sparsify (measure top-k attention mass concentration per block offset). If it is already near-uniform or
near-sparse, the KV-selection line is dead on arrival.

**Confidence: low as a method (no KV cache in the drafter), medium as a mechanism/explanation** for the
established Draft-OPD tree-rollout result — and it is the only one of the five that directly corroborates
"the rejected branch still matters".

---

## 5. FLy — Training-Free Loosely Speculative Decoding

**Citation.** ("FLy", authors not captured in the extracted text header) ICLR 2026; arXiv:2511.22972. Training-free
relaxation of the exact-match verification rule using the target's own entropy plus a deferred look-ahead.

**Genuinely new component.** A two-tier *verification* rule that adds zero forward passes, replacing FLy's own
Algorithm 1 steps 10–29. At the first mismatch position j (Δ_j = 0 in the standard match indicator):
- **Entropy gate.** Normalized target entropy `h_j = [−Σ_v p_{M_T,j}(v) log p_{M_T,j}(v)] / log|V|`.
  `Gate(j) = Strict if h_j < θ, else Defer`. Strict = standard SPD (reject from j onward); Defer = postpone.
- **Token-level deferred window of width W.** `N_W(j) = Σ_{i=j+1}^{j+W} (1 − Δ_i)`, and
  `DeferDecide(j) = Accept if h_j ≥ θ AND N_W(j) = 0 AND j + W ≤ K`, else Reject.
  Rationale, quoted: "When conditioned on a semantically invalid token, [the target] typically exhibits
  corrective behavior in its subsequent generations." Acceptance is retroactive — the mismatch is kept only if
  the target does not diverge again in the next W tokens. `s_FLy = min{s_gate, s_defer}`.
- Overhead is the entropy of logits already computed: 0.45 ms (L3 70B) / 0.57 ms (L3 405B) per round vs
  245.51 ms drafting and 58.62 ms verification. "FLy introduces no additional forward passes."

**What it needs.** Nothing but the verification logits. Draft length K=15 (70B) / 25 (405B); defaults W=6, θ=0.3;
no training, no verifier, no extra module. Assumption: the target's own hypothesis — that its continued
divergence is a reliable error signal — and a K large enough that j+W ≤ K for most mismatches. Trade-off stated
in its own Limitations: it preserves *semantic* consistency, not verbatim reproduction, so it is unsuitable
where exact token agreement is required.

**Applicability to GLANCE.** This is the most direct hit on a named open problem: **acceptance on high-entropy
text**, which is precisely where GLANCE fails. GLANCE §5.2 states the split is sharp: on grounded tasks the
head wins ("up to 7.6% faster than the production head"), but "Where the output is free-running text the
eight-pass chain leads instead … our deep candidates drift where a chain stays coherent by construction", with
captioning/TextVQA losses at one prompt in 101 vs at least 70 on grounded tasks. FLy's gate is driven by
exactly the scalar GLANCE already models: GLANCE fits `logit p_m(H) = b0 − b1 H` with pooled `b0 = 0.96`,
`b1 = 0.63` over 7751 rounds (captioning slope 0.46 → ChartQA 1.71). So the gate threshold can be *derived from
GLANCE's own published law* rather than taken from FLy's LLM-tuned default, and the target's full next-token
distribution at every tree node is already computed in the ancestor-masked pass — entropy is free at every node.
Two concrete, genuinely new components become available:
1. **Entropy-gated commit instead of longest-greedy-path commit** — a second, explicitly *lossy* operating point
   for GLANCE, with a measured accuracy-recovery number as the price.
2. **Tree-native deferred window** — FLy needs the target's behaviour over the next W tokens along the deferred
   path, which a sequential drafter cannot get without extra passes; GLANCE can get it by extending the *same*
   packed ancestor-masked pass with W extra rows along the candidate path (the pack already holds N=63 parallel
   rows, so this is not a new forward pass). That is a real algorithmic contribution, not a port.
The price is explicit and must be reported honestly: GLANCE's headline is losslessness (Theorem 1,
bitwise-identical to greedy). Adopting FLy means shipping a *second* mode, not improving the first.
**Calibration warning that is itself a finding:** FLy's default θ=0.3 is on `h = H_nats / log|V|`. For
Qwen3-VL-8B's ~152k vocabulary that is H ≈ 3.58 nats, while GLANCE measures target mean next-token entropy of
**0.26 nats (DocVQA) vs 0.72 nats (mismatched image)** — i.e. ≈0.02–0.06 normalized. Naively porting θ=0.3
means **the gate never fires**; θ has to be recalibrated by roughly an order of magnitude, and whether a
workable θ exists that helps captioning without hurting grounded tasks is the empirical question.

**Expected effect size, with regime caveats.** FLy's own Table 2/3 (Llama-3.1-70B/405B, 8B draft, K=15/25):
mean speedup 2.53×–5.21× on OOD with τ up to 17.13–18.89, vs EAGLE-3 at 1.52–1.56× with τ 2.31–2.50; the
ablation is the informative one — disabling the gate (θ=0, defer everything) gives 2.98× / τ 12.76 / 97.7%
recovery, θ=0.3 gives 2.86× / τ 12.61 / **100%** recovery, and θ=1 (reject all mismatches ⇒ degenerate to
standard SPD) collapses to 1.64× / τ 9.89. So the gate+window is worth **1.74×** over strict verification in
their setting. Accuracy: "over 99% recovery score" across datasets and scales (Figure 3); cross-model pairs
1.85×–3.54× at 99.1–100% recovery. **Not transferable as magnitudes**: text-only, autoregressive drafters,
K=15–25-token chain drafts, a 405B target, temperature 0/1, AMD MI355X, and τ values of 11–19 — a *different
draft-length regime* from GLANCE's B=16 block with τ=5.16. Also FLy *loses* to EAGLE-3 on in-domain data
(L33 70B HumanEval: 2.75× vs 3.74×), so it is not a strict win even at home. **Not stated:** any multimodal,
VLM, or tree result.

**Cheapest falsifying experiment (<1 day, 2 GPUs, inference only).**
Pure logging first, ~2 hours, 1 GPU: in the existing GLANCE verification pass, at the first mismatch node of
each round record the normalized target entropy `h_j` (free — the row already exists), per task, on COCO
captioning / TextVQA / ChartQA. Then read two things off the distributions. **Falsified if the first-mismatch
entropy on the free-text tasks sits *below* any θ that ground tasks can also tolerate** — i.e. if no threshold
separates "captioning mismatches the target would bless" from "ChartQA mismatches the target would not", the
gate has no operating point and the whole line dies. Second stage, same day: implement gate-only acceptance
(skip the window) plus a W≤2 probe extension inside the packed pass, and measure τ and caption quality
(CIDEr/BLEU or an LLM judge) vs the frozen target's own greedy output on COCO. **Falsified if τ gain on
captioning < 5% or quality recovery < 99%.**

**Confidence: medium-high that the mechanism applies, medium that it publishes.** It attacks the exact named
failure mode, costs no extra compute, is inference-only and falsifiable in hours, and the tree-native deferred
window is a genuine new component that FLy could not build. The reasons for caution are real: it trades away
GLANCE's lossless headline (so it must be framed as a second operating point with a measured accuracy cost),
the entropy-scale mismatch means the tuning may not work out, and the two other lossy-verification papers here
(SelfJudge, FLy) are both ~99% recovery on *text*, which is not evidence for caption fidelity.

---

## Ranking — most promising publishable method improvement

1. **FLy — best.** It is the only paper that attacks the failure mode GLANCE names against itself (free-running
   text, where "our deep candidates drift where a chain stays coherent by construction"), the mechanism costs
   zero extra forward passes because the target's per-node distributions already exist in the ancestor-masked
   pass, GLANCE's own fitted entropy law `logit p_m(H) = 0.96 − 0.63H` gives the gate a principled threshold
   instead of FLy's LLM default, and "extend the packed pass by W rows along the deferred path" is a real new
   component rather than a port. Cheapest falsifier is ~2 hours of logging that can kill it outright. The
   honest cost: it forks GLANCE into a lossy mode and abandons Theorem 1 for that mode, and FLy's 1.74× gate
   benefit and 99% recovery are from text LLMs with 15–25-token chain drafts, so the magnitude is unproven here.
2. **KnapSpec — strong second.** Correct objective (tokens-per-time, not accepted length) for the tree budget
   allocation open problem, with a portable formal guarantee (Lemma 4.1) and a single-pass DP that fits
   GLANCE's "one packed target pass" structure. GLANCE's own §5.4 saturation report and "Growing N adds
   verifier FLOPs inside that single pass" are the opening, and the falsifier is a pure inference-time N sweep.
   Downgraded only because the transfer is a re-derivation, the likely gain is a few percent, and the head's
   layer-fusion cost is small relative to the target pass (unlike KnapSpec's setting).
3. **SelfJudge — third, and only as a training signal.** Not applicable as a verifier to a bitwise-lossless
   system — say it plainly. But its semantic-preservation score is a reimplementable, cheap (~minutes for a few
   thousand anchors, vs 8.5 h for full labeling), target-derived way to test the *specific* Draft-OPD puzzle
   "why does reverse KL on the first rejected position help, and why does OPD lose on long free text", by
   soft-weighting alpha_rev at semantically-preserving rejections. Explains more than it improves.
4. **Vegas — fourth.** The KV-selection method is inapplicable: GLANCE's block head has no prefix KV cache to
   sparsify, and its headline numbers are from 96K–128K text-context serving at 7% sparsity. Its value is
   evidentiary: it independently establishes that rejected draft tokens still describe the target's plausible
   continuations (3%–14% accepted tokens lost when discarded tokens are excluded), which is the best external
   support for Draft-OPD's tree-rollout result, and gives one concrete variant (probability-weighted alpha_rev).
5. **HIPPO — last, weakest.** Not applicable because GLANCE's drafter never reads raw visual tokens — it
   cross-attends to the target's fused vision-language states — so target-guided visual-token pruning, the
   paper's whole first half, has no referent in this system. The remaining piece (VPSD draft/verify overlap) is
   scheduling rather than method, and its ceiling is the head's share of round latency in a 1.05B-vs-8B pairing,
   versus HIPPO's own 7B-vs-72B video pair where the draft is 30.7–38.3% of round time — roughly an order of
   magnitude more favourable than GLANCE's.
