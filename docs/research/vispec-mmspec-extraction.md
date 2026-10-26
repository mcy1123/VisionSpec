# High-fidelity extraction: ViSpec (2509.15235) and MMSpec (2603.14989)

Scope note for both blocks: GLANCE reference regime = frozen **Qwen3-VL-8B** target + ~**1.05B** 5-layer block-diffusion (DFlash-family) draft head, B=16, one draft pass per block, prefix-closed tree budget 63 / branch 8, verified in one ancestor-masked target pass; reported ChartQA τ 5.16 (bonus-included) / 3.42× at budget 63. Sibling GLANCE-2B = Qwen3-VL-2B target + ~**210M** 5-layer head, B=8, 5 kept layers of a 28-layer text tower, hidden 2048; chain τ 2.83 → tree τ 3.44; decode-only 1.33× / 1.83× (b31 k6) / 1.96× (b63 k8); fp32 lossless audit passes; **open problems: SFT plateau at τ 2.85–2.89, drafter weak on long free-form text.**

**τ convention warning (applies to both blocks).** GLANCE defines `τ = E[a + 1] = E[a] + 1` (Technical eq. 1) and explicitly says systems that log the bonus-excluded count `E[a]` are "raised by the committed token so every row sits on one scale." ViSpec logs the **bonus-excluded** count (τ = "average number of tokens accepted from the draft model per drafting-verification cycle"); MMSpec's MAT is likewise an accepted-token count that can sit below 1. So ViSpec/MMSpec τ values must be **+1** to be comparable with GLANCE's τ.

---

# Paper 1 — ViSpec

## 1. Citation line

**ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding.** Jialiang Kang¹², Han Shu², Wenshuo Li², Yingjie Zhai², Xinghao Chen²† (¹Peking University, ²Huawei Noah's Ark Lab). arXiv:2509.15235v5 [cs.CV], 23 Oct 2025; footer reads "39th Conference on Neural Information Processing Systems (NeurIPS 2025)." One-line: an EAGLE-2-style single-layer feature-level drafter for VLMs whose draft path is fed a Q-Former-compressed image (default: **one** compressed embedding) plus a *global* visual feature added to every draft hidden state, trained on synthetic long responses elicited from the target VLM by prompt rewriting.

## 2. The genuinely new method component

Three components; none of them is a new decoding algorithm — the decoding machinery is EAGLE-2's, reused verbatim.

**(a) Image-embedding compression via a vision adaptor.** Motivation is a stated theoretical argument that a shallow draft model cannot filter redundant image patches. Setup quoted verbatim: "Consider a sequence of R + 1 image and text embeddings e₁, …, e_{R+1} ∈ ℝᵈ, where R embeddings are identical, i.e., e_{r1} = … = e_{rR} = s, and a unique token at position u has embedding e_u = t." Attention output (verbatim):

> `y_i = Σ_{j=1}^{R+1} α_ij v_j,  where  α_ij = exp(q_i k_j^⊤) / Σ_{k=1}^{R+1} exp(q_i k_k^⊤),`   (Eq. 1)

> `α_iu = exp(B) / (R exp(A) + exp(B)),`   (Eq. 3)

> `y_i ≈ Σ_{m=1}^{R} v_r^m / R = W_v s,`   (Eq. 4)

i.e. "As R → ∞, the denominator is dominated by R exp(A), causing α_iu → 0 … effectively averaging over the redundant tokens and neglecting the unique token."
The module itself: "a lightweight Q-Former-inspired [16] vision adaptor (see Fig. 3). This module utilizes a lightweight Transformer encoder with a fixed set of learnable query vectors. The visual features extracted from the input image serve as key and value inputs to the Transformer's attention layers, while the learnable query vectors function as queries." Figure 3 caption: "A compact Transformer encoder with fixed learnable query vectors q processes input visual embeddings v through an attention layer, yielding a small set of compressed image tokens c and a single global visual feature g." The figure depicts one attention layer + a linear projection; **the number of adaptor layers, the query dim, and the compressed-token width are not stated** (MMSpec's reimplementation config lists `num_q=2`, while ViSpec's own ablation adopts 1 — see §5).

**(b) Global visual feature injection** — the component with the clearest reimplementation recipe. The global vector is "derive[d] … from the final output of the vision adaptor module … transformed and incorporated into the hidden states of all subsequent text tokens in the draft model." Verbatim:

> `f_t^aug = f_t + W_g g,`   (Eq. 5)

"where f_t is the original hidden state, g denotes the global visual feature vector, and W_g is a learned projection matrix." One matrix, one add per token, no extra forward pass. It is claimed to fix the lost-in-the-middle effect for a shallow drafter over long responses.

**(c) Training-signal construction and the anti-shortcut argument.** Data: "we modify prompts in datasets such as visual question answering or image captioning to elicit longer, more descriptive responses from the target VLM" — concretely, "we append the prompt 'Please answer with at least 1000 words.' to each sample to elicit long responses." Loss (verbatim):

> `L = CrossEntropy(p_i, p̂_i),`   (Eq. 6)

"where p_i and p̂_i denote the target model's and draft model's probabilities for the i-th token, respectively." The paper's Figure-4 argument is that training on **greedy** target responses creates a one-to-one f↔e correspondence and "effectively reduc[es] to a single-step Medusa [2]"; ViSpec instead uses target **sampling** to "disrupt one-to-one correspondences between hidden states f and embeddings e" and feeds the drafter's own hidden states f′ back in for self-correction.

**Decoding algorithm (not new):** "We enhance this approach by adopting the context-aware dynamic draft tree from EAGLE-2 [18]." Implementation: "We implement a single-layer draft model that mirrors a decoder layer of the target model … During inference, we adopt EAGLE-2's draft tree structure, configuring 30 draft tokens, a tree depth of 3, and selecting 8 nodes during expansion, applied consistently across all models and tasks."

**Sizes when stated** (Table 4, "Params (M) / GFLOPs / Latency (s), Base → +Adap"):
- LLaVA-1.6 7B: 367 → 451 M; 956 → 179 GFLOPs; 0.227 → 0.231 s
- LLaVA-1.6 13B: 534 → 665 M; 1460 → 279 GFLOPs; 0.334 → 0.334 s
- Qwen2.5-VL 3B: 404 → 425 M; 57.3 → 18.3 GFLOPs; 0.002 → 0.004 s
- Qwen2.5-VL 7B: 826 → 890 M; 172 → 55.5 GFLOPs; 0.018 → 0.016 s

So the adaptor costs **+21M to +131M** params depending on target; the drafter itself is 0.23–0.89B (1 layer + embeddings + LM head), **not** a small standalone LM. Hidden width / attention heads of the drafter: **not stated** (inherited from the target, since it "mirrors a decoder layer of the target model").

## 3. What it needs

- **Training required.** Two-stage, supervised: ShareGPT first ("68,000 dialogue iterations") then multimodal fine-tuning on "68,000 samples randomly selected from the LLaVA Visual Instruct Pretrain LCS dataset," for ViSpec augmented with synthetic long responses generated by the target. Hyperparameters: "learning rate of 3e-6, a batch size of 8, and the AdamW optimizer … trained for 20 epochs with a 1-epoch warmup and linear learning rate decay, supporting a maximum sequence length of 2048." Draft models trained on **8 GPUs**; all experiments run on **a single GPU**.
- **Extra module:** the vision adaptor (params above) plus a learned W_g. Because the adaptor emits both c (compressed tokens, prepended to the draft input) and g (global feature), the drafter is a modified-input model, not a drop-in head.
- **Access to the target's vision encoder**: the adaptor consumes the VLM's original vision-encoder features ("We modify their input pipelines to process image patch embeddings from the VLM's original vision encoder"), so this is *raw visual features*, not fused hidden states.
- **Training-signal requirement (a hard one, and the interesting one):** a multimodal corpus with long assistant responses, which "are scarce in the public domain" — hence the synthetic generation step that needs the target VLM at inference time with a long-answer prompt.
- **Assumption that its own numbers contradict-ish:** "As ViSpec employs strict speculative decoding to preserve the target model's generation quality, quality evaluation is unnecessary." It also states it adopts EAGLE-2's acceptance condition ("we adopt the same acceptance condition as EAGLE-2" — said of Medusa, but ViSpec inherits the EAGLE-2 tree and its typical-acceptance rule too). GLANCE's audit found ViSpec does **not** reproduce greedy text (see §5). No accuracy table is reported anywhere in the paper.

## 4. Applicability to GLANCE / GLANCE-2B

Split verdict — one component is the highest-value transferable idea found in either paper; the core visual mechanism is largely **not applicable**.

- **NOT applicable — image-embedding compression (Eqs. 1–4 + adaptor).** The entire motivation is that a shallow drafter drowns in *raw, redundant visual tokens*. GLANCE's head never reads raw visual tokens: it reads the target's already-fused vision-language hidden states at 5 kept layers, i.e. the redundancy-filtering the ViSpec adaptor exists to perform has already been done by the target's own 36/28 layers. So there is no redundant-token dilution to fix. The remaining sub-question is cost: GLANCE's one-pass draft is measured at **4.6 ms vs 24.5 ms for an AR decode forward (0.19×)** on Qwen3-VL-8B, while the packed tree verify is 26.3 ms (1.07×) — the round is verify-dominated, so compressing what the drafter attends to has ≤19% of a forward as its total headroom. Also relevant: GLANCE's own ablation shows swapping fused vision-language states for a text-only LLM's "costs almost nothing on captioning … and more as the task becomes grounded, to a minimum of 80% on DocVQA" — visual conditioning is worth keeping, and the fused states are how GLANCE keeps it.
- **PARTIALLY applicable — global visual feature injection (Eq. 5).** The gap it targets (visual grounding decaying as the response lengthens, "lost in the middle" in a shallow drafter) is structurally present in a 5-layer 210M block head that must hold visual grounding across a 16-position block and a long free-form response. In GLANCE the global vector does not need raw visual tokens: mean-pool the target's kept hidden states at image-token indices (already materialized, zero extra target compute) and add `W_g g` to the drafter's text hidden states at every offset. Cost: 2048×2048 ≈ **4.2M params (+2% on 210M)** plus one add per offset — within noise of the draft-pass cost. This is the one ViSpec mechanism that maps onto GLANCE almost unchanged.
- **APPLICABLE — the training-signal recipe (highest value).** This is the component that speaks to GLANCE-2B's stated open problems ("acceptance saturates — SFT plateaus at τ 2.85–2.89" and "drafter weak on long free-form text"). ViSpec's fix is not architectural: elicit long responses from the target by prompt rewriting, then train on them. Its own ablation attributes the **largest single jump** to exactly this step (see §5: τ 2.14 → 3.30 on COCO Captions, +54%). ViSpec's diagnosis of *why* is also a direct challenge to GLANCE's stated training protocol: GLANCE trains on the target's own **greedy** generations, and ViSpec argues (Fig. 4) that greedy target traces make the drafter's hidden-state input and token input one-to-one, degenerating the head to single-step Medusa; sampling breaks that correspondence. **This is a falsifiable claim about GLANCE's plateau, and it is cheap to test.**
- **Does NOT address:** drafting parallelism (ViSpec still pays 3 sequential draft expansions / 30 draft tokens vs GLANCE's single pass), memory (adds params, removes none), drafter capacity-vs-cost (ViSpec's drafter is 1 layer; GLANCE-2B's is 5 and already capacity-limited), or τ on grounded short-answer tasks (ViSpec's weakest results are on the shortest/most gaze-anchored tasks).

## 5. Expected effect size, and whether the regime is comparable

**Flag:** every headline number below is from **different targets, a different drafter family (1-layer EAGLE-style + EAGLE-2 tree, depth 3 / 30 tokens / top-8), a different tree budget (30 vs GLANCE's 63), and a **bonus-excluded** τ convention**. None is measured on Qwen3-VL-2B or Qwen3-VL-8B.

ViSpec's own Table 1 (temperature 0, bonus-excluded τ / speedup, 8 datasets, avg column):

| Target | Medusa | EAGLE-2 | ViSpec |
|---|---|---|---|
| LLaVA-1.6-7B | τ 0.72 / 1.42× | τ 1.31 / 1.62× | **τ 2.98 / 2.58×** |
| LLaVA-1.6-13B | τ 0.82 / 1.48× | τ 1.89 / 1.86× | **τ 2.89 / 2.38×** |
| Qwen2.5-VL-3B | τ 0.61 / 1.14× | τ 1.16 / 1.39× | **τ 2.19 / 1.87×** |
| Qwen2.5-VL-7B | τ 0.60 / 1.11× | τ 1.34 / 1.40× | **τ 2.24 / 1.80×** |

At temperature 1 the Qwen targets collapse to ViSpec τ 1.42 / 1.47× (3B) and τ 1.66 / 1.49× (7B) — i.e. **sampling roughly halves acceptance**, which matters if GLANCE-2B is ever audited under sampling.
Best case quoted: 2.90× on TextVQA (LLaVA-1.6-7B, T=0); paper-wide range "1.37× to 3.22×" vs EAGLE-2 "1.06× to 2.17×" and Medusa "0.95× to 1.67×". High-resolution appendix: Qwen2.5-VL-7B HR-Bench 4K τ 2.16 / 1.29×, MME-RealWorld τ 2.11 / 1.37×. Video, zero-shot transfer: MSVD-QA τ 2.16 / 1.46×, MVBench τ 2.09 / 1.32×.

**Component ablation (Table 3; values match the LLaVA-1.6-Vicuna-7B rows exactly although the caption does not name the model), τ / speedup on COCO Caps / GQA / MME:**

- baseline (EAGLE-2): 1.24 / 1.80× · 1.74 / 1.64× · 1.25 / 1.68×
- + image embedding compression: 2.04 / 2.37× · 2.15 / 1.92× · 2.04 / 1.83×
- + global visual injection: 2.14 / 2.42× · 2.25 / 2.03× · 2.14 / 1.95×
- + dataset generation: **3.30 / 3.22× · 2.88 / 2.22× · 2.84 / 2.55×**

So: compression does most of the early work (+65% τ on COCO), global injection adds a small consistent step (**+7% speedup**, +0.10 τ on COCO), and the long-response data recipe adds the rest (**+30% speedup, +1.16 τ on COCO**). Compressed-embedding count (Table 2, same model): 1 → τ 3.30/3.22×; 4 → 3.24/3.24×; 16 → 3.23/3.21×; 64 → 3.25/**2.71×** on COCO Caps, i.e. **more compressed tokens buy no τ and cost speedup** — the paper's "a single compressed image embedding adequately captures essential visual information" conclusion.

**Accuracy deltas: none reported** ("quality evaluation is unnecessary"). Any accuracy claim for ViSpec is an assumption, not a measurement.

**Critical external cross-check that must temper all of the above.** GLANCE (arXiv 2609.00355) re-ran ViSpec as a baseline under its own matched protocol, and reports (its Table 2, τ is **bonus-included**):
- On GLANCE's target Qwen3-VL-8B, with ViSpec's head trained by "the authors' official pipeline": drafter 0.23B, τ = caption 1.72 / TextVQA 2.00 / InfographicVQA 1.39 / DocVQA 1.86 / ChartQA 1.61, geomean speedup **1.05×**, lossless column **×**.
- On ViSpec's own home target Qwen2.5-VL-7B, ViSpec's **released** head: 0.35B, τ = 3.34 / 3.26 / 3.21 / 3.04 / 3.59, **1.76×**, lossless **×**. (Subtract 1 for ViSpec's convention: caption τ 2.34 vs ViSpec's own COCO-Caps 2.27 — the two independent measurements agree closely, which is a good sign for both.)
- GLANCE's audit: "GLANCE reproduces greedy decoding on every audited prompt, while ViSpec and Medusa reproduce none under any tree setting, placing their divergence in the relaxed acceptance rule."

**Bottom line on comparability:** ViSpec's *method-level* numbers are not the draw — under GLANCE's own matched harness ViSpec is at 1.05× geomean and is not lossless, versus GLANCE's 2.49× (bonus-included τ 3.04–5.16). Its *component*-level numbers (global injection +7% speedup; long-response data +30% speedup) are measured on LLaVA-1.6-7B, a much easier target (τ 2.98 vs 2.24 on Qwen2.5-VL-7B), so treat +30% as an upper bound on a Qwen-family target.

## 6. Cheapest falsifying experiment on GLANCE-2B (2 GPUs, <1 day)

**Experiment A (primary; targets the SFT plateau and the long-free-form weakness).** Take the frozen 210M 5-layer head at its plateau (τ 2.85–2.89). Build two SFT corpora from the same prompt pool, ~5–10k samples each: (i) **greedy traces** (existing recipe), (ii) **sampled traces** — target sampling at T = 0.7, plus ViSpec's prompt rewrite ("Please answer with at least 1000 words" / "Provide a detailed description of the given image") to force long responses. Fine-tune two heads for a **fixed, equal step budget** (a few hours each on 1 GPU; the 2nd GPU runs the eval). Measure, on a held-out split that is long-free-form by construction (captioning + DocVQA/ChartQA), at the existing b63 k8 tree: τ, per-**offset** acceptance (offsets 1…8), and decode-only speedup.
- **Falsifies ViSpec's training argument if** sampled-data τ ≤ greedy-data τ (Δτ ≤ +0.05) and the per-offset acceptance curve is unchanged — i.e. the plateau is a capacity/data-*volume* wall, not a correspondence-degeneracy wall. ViSpec's Fig. 4 claim then simply does not hold for a block-diffusion head that predicts all offsets in one pass.
- Secondary read-out from the same run: ViSpec's own claim is that prompt-rewritten long data helps *most* — if the long-data head does not beat the plateau specifically on captioning/long-form while matching elsewhere, the "weak on long free-form text" gap is not data-limited.

**Experiment B (secondary, ~half a day, and cheap enough to run alongside).** Implement Eq. 5 exactly: `f_t^aug = f_t + W_g g`, with `g` = mean-pool of the target's kept hidden states at image-token indices (already produced by the training/verify pass — no extra target compute), `W_g` zero-initialized 2048×2048. Warm up from the plateau checkpoint for a short fixed budget; measure Δτ on a grounded set (ChartQA) vs a free-form set (captioning), plus per-offset acceptance.
- **Falsifies if** Δτ < 0.05 on both sets, or if the gain is flat across offsets 1→8 — either would mean the "lost-in-the-middle across the block" mechanism ViSpec posits does not bind at B=8 with 5 kept layers.

**Experiment C (diagnostic, hours, no training) — the one that also serves the MMSpec block.** Log the drafter's attention mass on image-token positions as a function of block offset, and log the target's image-attention ratio per step per MMSpec's `S_t = max_i A_t^(i)`. If attention mass on visual positions is already non-decaying across offsets 1→8, ViSpec's global-injection motivation and MMSpec's VisSkip gate both lose their premise for GLANCE-2B before any training is spent.

**Cannot be tested cheaply:** the adaptor/compression branch (Eqs. 1–4) — it presumes a drafter that ingests raw visual tokens, which GLANCE-2B does not; retrofitting one to test the redundancy argument would be a new drafter, not a one-day experiment.

## 7. Confidence

**Medium** that ViSpec *transfers*; **low** that ViSpec-the-method helps; **medium-high** for the training-signal component specifically.

- Mechanism-level: Eq. 5 is a 2-line change with a 2% parameter cost on a 210M head and no new input pipeline — the reimplementation risk is near zero, so any measured Δτ is trustworthy.
- The gap it targets (long free-form weakness, plateau) is the *right* gap, and its data-recipe ablation is the largest effect in the paper. But that ablation is on LLaVA-1.6-7B with a 1-layer EAGLE drafter, on COCO Captions/MME/GQA — not a Qwen target, not a block-diffusion head, not a grounded-chart task.
- The negative evidence is strong and specific: GLANCE already implemented ViSpec faithfully on its own target and got τ 1.72 caption / 1.61 ChartQA and 1.05× geomean (bonus-included) with a lossless-audit failure. There is **no** headroom in copying ViSpec wholesale.
- Two things would move this to high: (i) if Exp. A shows the plateau is broken by sampled/long-response data, ViSpec's most transferable idea is confirmed on GLANCE-2B directly; (ii) if Exp. C shows visual attention decays across block offsets.

---

# Paper 2 — MMSpec

## 1. Citation line

**MMSPEC: Benchmarking Speculative Decoding for Vision-Language Models.** Hui Shen¹,*, Xin Wang²,*, Ping Zhang²,*, Yunta Hsieh¹, Qi Han³, Zhongwei Wan², Ziheng Zhang², Jingxuan Zhang⁴, Jing Xiong⁵, Ziyuan Liu⁶, Yifan Zhang¹, Hangrui Cao⁷, Chenyang Zhao⁸, Mi Zhang² (¹University of Michigan, ²The Ohio State University, ³Independent, ⁴Indiana University, ⁵HKU, ⁶Peking University, ⁷Carnegie Mellon University, ⁸LMSYS Org). arXiv:2603.14989v1 [cs.CV], 16 Mar 2026, marked "Preprint." One-line: the first standardized benchmark + unified evaluation platform for speculative decoding on VLMs — 600 multimodal samples in six task categories, ten algorithms, two 7B targets, four A100s — plus a plug-and-play gating method (ViSkip) derived from the diagnostic findings.

## 2. The genuinely new method component

**Two deliverables. Neither is a drafter architecture; there is no new draft model, no new head, no new training objective.**

**(a) The benchmark/eval platform (the actual contribution).**
- **600 samples, six categories, 100 each**, "7 sources": General VQA = GQA; Text VQA = TextVQA; Image Captioning = COCO; Chart VQA = CharXiv; Complex Reasoning = MMMU-Pro; Multi-turn Conversation = ConvBench + MM-MT-Bench. Average output lengths computed with Qwen2.5-VL-7B (Table 1): GQA 46.98, TextVQA 63.15, COCO/IC **191.90**, CharXiv/CQA 68.56, MMMU-Pro/CR **285.60**, MTC **747.65**; total average 233.97 tokens. (This is the property that matters for GLANCE-2B: the set is deliberately weighted toward long outputs, not short VQA.)
- **Four design principles, quoted:** "(i) workload diversity, (ii) balanced topic coverage, (iii) explicit multi-turn support, and (iv) method-agnostic measurement."
- **Ten algorithms**, split training-based (EAGLE-1, EAGLE-2, EAGLE-3, Medusa, MSD, ViSpec) and training-free (SAM Decoding, Lookahead, Recycling, PLD), each characterized by key idea / linear-vs-tree drafting structure / whether it reuses repetitive content / vision-aware vs vision-agnostic.
- **Protocol:** "Since the selected speculative decoding methods are all lossless … our benchmark only contains two key efficiency metrics": **MAT** ("Mean Accepted Tokens … the average number of tokens accepted per speculative decoding step") and **Speed** ("Walltime Speedup Ratio … relative to vanilla autoregressive decoding"), plus per-request latency CDFs. Setup: Qwen2.5-VL-7B-Instruct and LLaVA-1.5-7B, "four NVIDIA A100 GPUs," batch size 1, max new tokens 1024, "Decoding temperatures t ∈ {0.0, 0.6}", method hyperparameters as in Table 6 (e.g. `EAGLE-2: depth=3, top_k=8, total_token=30, threshold=0.3`; `ViSpec: depth=3, top_k=8, total_token=30, num_q=2`; `MSD: depth=5, top_k=10`).
- **Appendix diagnostic probe (`A.3`)** — the most reusable piece: "We run MSD, ViSpec, and SAM while logging, for every decoding step, both the speculative acceptance length and the visual-attention ratio computed by the same attention-scoring rule used in ViSkip. We then define a high-visual step as one whose visual-attention ratio is at least 0.35."

**(b) ViSkip — a plug-and-play gating rule (small, fully specified).** "a vision-aware speculative decoding framework that adaptively alternates between conventional decoding and speculative drafting based on the estimated visual relevance of the current token state." Verbatim equations:

> `A_t = CrossAttn(h_t, V),`   (Eq. 1)
> `S_t = max_{i∈[1,M]} A_t^{(i)},`   (Eq. 2)
> `I_vision(t) = 1 if S_t > τ, 0 otherwise.`   (Eq. 3)
> `Decode(t) = SpecDecode(F, D, K) if S_t ≤ τ;  Greedy(F) if S_t > τ.`   (Eq. 4)

Algorithm 1, verbatim: `Require: Full model F, draft model D, threshold τ, draft length K` / `1: for each decoding step t do` / `2: Compute cross-attention scores A_t` / `3: Compute S_t = max_i A_t^(i)` / `4: if S_t ≤ τ then` / `5: Draft up to K tokens using D` / `6: Verify with F (standard speculative decoding)` / `7: else` / `8: Generate one token using F` / `9: end if` / `10: end for`.
**Stated threshold: τ = 0.35** ("which matches the ViSkip threshold used in our probe runs"). **Number of draft layers/heads, hidden width, parameter counts: not stated** — ViSkip adds no parameters (it gates an existing drafter), which is its selling point. Note the paper promises "Due to the page limit, we measure the impact of τ on the performance of VISKIP in the Appendix" — **the extracted appendix contains no τ sweep**; the only τ value in the paper is 0.35 in the probe, and that value is described as the *probe's* threshold, not necessarily the main run's. The draft length K required by Algorithm 1 is **not stated** for ViSkip's own runs (Table 6 lists method-specific parameters for the ten baselines only — there is no ViSkip row).

**Two specification defects worth naming before anyone reimplements this.** (i) `CrossAttn(h_t, V)` is not a thing either evaluated target has: Qwen2.5-VL and LLaVA-1.5 place image tokens *in the decoder sequence*, so there is no decoder→vision cross-attention; what must actually be meant is self-attention mass from the current text position onto image-token positions. The paper never says which layer, which heads, or how heads are reduced. (ii) Eq. 2 (`S_t = max_i A_t^(i)`, a single maximum softmax weight) and the appendix's wording ("visual-attention ratio", a normalized share) are not the same statistic, and a single weight of 0.35 versus 35% total image attention are very different operating points. Also note Table 7's "MMSPEC testmini set" is never defined anywhere else in the paper.

## 3. What it needs

- **Training-free**, both the benchmark (obviously) and ViSkip. ViSkip needs only the decoder hidden state h_t and cross-attention weights to visual tokens at each step, plus an existing drafter and a threshold. No new parameters.
- **Assumptions/limitations it states or implies:** (i) the ten methods are assumed lossless, so **no accuracy is measured anywhere** — this is an assumption, not a measurement, and it is contestable (GLANCE's fp32 audit found ViSpec and Medusa reproduce greedy decoding on **none** of its audited prompts; EAGLE-2 and SAM also run relaxed acceptance, with `threshold=0.3` and `threshold=1.0` respectively in MMSpec's own configs); (ii) the cross-attention weights must be exposed by the engine, which is a real systems requirement (asking for attentions in a CUDA-graphed serving path is not free); (iii) the gate makes the decode policy input-dependent, which perturbs batching/scheduling behavior; (iv) results are at **batch size 1** in the main table (batch sensitivity is figure-only).
- **Cost:** one extra scalar per step plus a branch; the `Greedy(F)` branch is a full AR token, so ViSkip can *lose* time wherever the draft would have been accepted.
- Benchmark cost: 600 samples × 2 models × 10 methods × 2 temperatures on 4 A100s — modest, which is exactly its point.

## 4. Applicability to GLANCE / GLANCE-2B

This is a benchmark paper, so the question is the protocol and the diagnostics, and both are reusable. Findings, with the numbers:

**(i) Concrete rankings (Table 3, greedy, Qwen2.5-VL-7B, overall MAT / Speed):** MSD **2.57 / 2.58×** (best) > EAGLE-1 2.36 / 2.11× > EAGLE-2 1.78 / 2.02× > ViSpec **1.29 / 1.51×** > Medusa 0.80 / 1.49× > EAGLE-3 **0.24 / 0.96×** (below AR) ; training-free: SAM 0.23 / 2.17×, Lookahead 0.33 / 1.07×, PLD 0.17 / 1.05×, Recycling 0.11 / 1.04×.
On **LLaVA-1.5-7B**: MSD 3.66 / 2.38×, ViSpec 3.00 / 2.58×, PLD 0.60 / 1.77×, Recycling 0.88 / 1.54×, Medusa 1.30 / 1.26×, EAGLE-1 0.76 / 1.22×, EAGLE-2 0.58 / 1.20×, SAM 0.38 / 1.13×, Lookahead 0.52 / 1.17×, EAGLE-3 0.04 / **0.70×**.
**Who loses and why, in the paper's words:** model-free methods are near-useless ("Lookahead achieves only 1.07× speedup on Qwen2.5-VL-7B"); text-only-designed trained methods degrade (EAGLE-2 only 2.02×); even vision-aware trained methods are unstable across tasks (MSD "1.80× to over 4×"; ViSpec "speedup fluctuates notably across subtasks").

**(ii) Per-category breakdown — read MAT, not speedup, and read it carefully.** For Qwen2.5-VL-7B, ViSpec's MAT is **2.01 (GQA, 47 tok) / 1.26 (TextVQA, 63 tok) / 1.75 (IC, 191.9 tok) / 1.57 (CQA, 68.6 tok) / 1.28 (CR, 285.6 tok) / 1.15 (MTC, 747.7 tok)** — its two worst cells are TextVQA (short and image-anchored) and MTC (longest); the avg-output-length column does not order the results. MSD is far flatter and much better: 2.32 / 2.37 / 2.35 / 2.67 / 2.42 / 2.51. The subagent's first-pass reading, "long free-form wins, short grounded VQA loses," is true of the **speedup column** (MSD 4.06× on CR vs 1.80× on TextVQA; ViSpec 1.12× on TextVQA) but that column is confounded: speedup = AR-time/SD-time, and reasoning/multi-turn tasks have a *larger* per-token AR cost, so a ratio computed per task mechanically flatters them. **The portable metric is MAT/τ, and on MAT there is no simple "longer is better" law** — on LLaVA-1.5-7B ViSpec's best cell is Image Captioning at **3.44 MAT / 2.83×** while MTC is 3.01, and on Qwen MTC is its worst. For GLANCE-2B the actionable version of this finding is: *do not conclude anything about long-form vs grounded from MMSpec's speedup column; treat the MAT-by-category table as the hypothesis and re-measure it.* GLANCE already takes the right position here ("Acceptance length τ of Equation (1) counts tokens, not time, so it does not depend on the engine and compares systems directly").

**(iii) Temperature.** "sampling at t = 0.6 generally makes speculative decoding less stable, with a clearer degradation for training-based methods." MSD 2.57/2.58× → 1.87/1.82×; EAGLE-1 2.36/2.11× → 2.09/1.76×; on LLaVA-1.5-7B "ViSpec decreases from 3.00/2.58× to 1.45/1.68×" and EAGLE-1/2 "collapse substantially." Training-free methods are lower-ceiling but temperature-insensitive.

**(iv) Batch size (Figure 4; numeric batch values are not stated in the text — figure only).** Two claims: "vision-aware speculative decoding methods, such as MSD and ViSpec, consistently achieve the highest speedups … MSD reaches around 2.3×–2.6× speedup on both Qwen2.5-VL-7B and LLaVA-1.5-7B across all batch sizes"; and "the speedup of methods such as EAGLE-2 and Lookahead decreases significantly when moving from small batch sizes to larger ones." Takeaway 2: "Vision awareness becomes increasingly important for speculative decoding in vision-language models, especially under larger batch sizes."

**(v) Throughput vs latency (Figure 5, latency CDFs; no latency numbers are given, only curves).** "some methods that achieve relatively high throughput speedups do not always exhibit the best latency performance in the CDF curves … they can still suffer from unstable behavior across samples, leading to longer latency for certain inputs." Takeaway 3: evaluate on both. This is a real caution for GLANCE, which reports mean decode ms/token and a paired bootstrap on a *mean* while shipping per-round commits whose length is heavy-tailed (P(a≥8) is reported to rise with grounding).

**(vi) The step-level probe (Table 7) — the single most reusable diagnostic.** On Qwen2.5-VL-7B over the MMSpec testmini set, using the 0.35 high-visual threshold: MSD 2170 measured steps, 332 high-visual (15.3%), avg accept **2.12 (high) vs 2.27 (low)**; ViSpec 1874 steps, 170 high-visual (9.1%), **2.12 vs 2.39**; SAM 2469 steps, 176 high-visual (7.1%), **0.14 vs 0.44**. Takeaway, verbatim: "When VLM attends more strongly to image tokens, speculative acceptance becomes worse, often collapsing to zero for visually grounded words or phrases."

**Does MMSpec address GLANCE's listed gaps?** Largely no, on the gaps as stated, and yes on *measurement*. It does not touch drafter capacity-vs-cost, visual-token handling inside the drafter, drafting parallelism, training-signal scarcity, or memory. It does not solve the acceptance plateau — its answer to "acceptance is bad where it matters" is to *stop drafting there*, which lowers accepted length by construction and trades it for wall-clock. What it gives GLANCE-2B is (a) an external, target-independent yardstick (MAT + walltime + per-request latency CDF across six long-output-weighted categories) that would expose whether τ 3.44 is good or merely self-consistent, and (b) the high-visual/low-visual acceptance split as a diagnostic for **why** the 210M head plateaus.

## 5. Expected effect size, and whether the regime is comparable

**Flag:** MMSpec's targets are **Qwen2.5-VL-7B-Instruct and LLaVA-1.5-7B** — 7B, and neither is Qwen3-VL-2B. Its drafters are 1-layer EAGLE-family heads, Medusa heads, MSD's multimodal drafter, and training-free heuristics — **no block-diffusion drafter, no B=8/B=16 block, no budget-63 tree**. Its MAT is bonus-excluded (≈ GLANCE τ − 1). Its τ values are therefore *not* directly comparable to GLANCE-2B's τ 2.83/3.44; only the **normalized comparison to the baselines it re-ran on the same target** is.

- Best available anchor for GLANCE-2B: **Qwen2.5-VL-7B (a Qwen2.5-family, not Qwen3, target at 7B)** — MSD 2.57 MAT / 2.58×; ViSpec 1.29 / 1.51×. Converting GLANCE-2B's τ to the bonus-excluded convention (τ − 1): chain 1.83, tree **2.44** at b63 k8. That places GLANCE-2B's block-diffusion head just under MSD's 7B MAT of 2.57 while using a ~210M head on a ~3.5× smaller target — a favourable but **not strictly comparable** prior, since MMSpec's drafters are 1-layer/chain-and-tree EAGLE-family heads at depth 3/30 tokens (MSD depth 5/top-10), not a one-pass B=8 block with a 63-node tree, and MMSpec gives no engine.
- **ViSkip's effect: no numbers reported.** Its entire evaluation is Figure 6 — "integrating VISKIP consistently improves the speedup of all three methods across different batch sizes," CDF curves shift left, and the latency breakdown shows reduced target-model time. **No speedup value, no MAT value, no accuracy value is stated anywhere for ViSkip**, and the promised τ sweep is absent.
- **Accuracy deltas: not stated** for any method — the benchmark assumes losslessness ("Since the selected speculative decoding methods are all lossless"), so no accuracy column exists. Given that ViSpec's own tree config carries `threshold=0.3` (EAGLE-2's relaxed/typical acceptance) and SAM's carries `threshold=1.0`, and GLANCE's fp32 audit found ViSpec and Medusa reproduce greedy on none of its audited prompts, this assumption is the benchmark's main validity hole. No byte-identity check is reported.
- **HARD RED FLAG — the speedup column is not trustworthy for the training-free methods.** Write `a` for MAT (accepted draft tokens per round, bonus-excluded, which is the only convention under which most rows are self-consistent). Then one round emits `a + 1` tokens, and since a round always costs more than one AR forward, the identity `speedup = (a+1) × (AR forward time / round time)` forces **`speedup < MAT + 1`** — a hard arithmetic bound. Table 3 violates it in **2 of 20 overall rows** (Qwen2.5-VL-7B SAM: MAT 0.23 / **2.17×**; LLaVA-1.5-7B PLD: MAT 0.60 / **1.77×**) and in **7 per-task cells** (SAM Qwen CR 0.19 / **6.53×**, SAM Qwen CQA 0.51 / 2.04×, SAM Qwen MTC 0.27 / 2.04×; Recycling LLaVA TVQA 0.19 / 1.87×, Recycling LLaVA MTC 1.00 / 2.37×; PLD LLaVA MTC 0.48 / 2.67×, PLD LLaVA TVQA 0.03 / 1.14×). Every violation sits in the **training-free** block; all ten training-based and all AR-baseline rows are self-consistent. So the MAT column is usable and the training-free speedups (especially SAM's) are artifacts of a different measurement — **do not size anything on MMSpec's speedup column for training-free methods.**
- Notable internal cross-checks worth carrying: MMSpec's ViSpec row uses `num_q=2` compressed queries while ViSpec's own ablation concludes **1** is optimal (`more compressed tokens buy no τ and cost speedup`, Table 2); and MMSpec's ViSpec IC MAT on Qwen2.5-VL-7B is 1.75 where ViSpec's own paper reports COCO-Caps τ 2.27 on the same target under its own prompts. Different prompt sets, different numbers — the benchmark is measuring the *method as configured*, not the paper's best case. Whether MMSpec retrained every baseline drafter or reused released checkpoints is **not stated** ("We followed the default parameters as specified in their original implementations"), nor is which engine was used (vLLM/SGLang/HF all unstated), nor how four A100s are used at batch size 1, nor the batch-size grid behind Figure 4.

## 6. Cheapest falsifying experiment on GLANCE-2B (2 GPUs, <1 day)

**Experiment D (primary, ~half a day, no training, and it is a measurement, not a method).** Port MMSpec's two-protocol ideas onto the existing GLANCE-2B eval:
1. **Re-score on a MMSpec-shaped slice**: take ~100 samples per category in MMSpec's proportions (or the actual MMSpec testmini set) through the existing Qwen3-VL-2B + 210M head at b63 k8, and report **MAT (bonus-excluded, so MMSpec-comparable), walltime speedup, and a per-request latency CDF** — not just mean ms/token. GLANCE-2B currently reports chain/tree τ and decode-only speedups on its own task set; this adds the only two things it lacks, an external yardstick and a tail-latency view.
2. **Run the A.3 probe**: for every decoding step, log `S_t = max_i A_t^(i)` over image-token positions and split accepted length into a **high-visual bin (S_t ≥ 0.35)** and a low-visual bin.
- **Falsifies ViSkip's premise (and the whole "acceptance dies on visually grounded tokens" story) for GLANCE-2B if** the high-visual bin's mean accepted length is within ~10% of the low-visual bin's — i.e. no ≥0.2-token gap of the MSD (0.15) / ViSpec (0.27) / SAM (0.30) magnitude. That is a one-page result and it directly determines whether any gating work is worth doing.
- **Also falsifies/bounds the throughput-vs-latency concern if** the per-request latency CDF for the b63 k8 tree is not heavier-tailed than the AR baseline's; if it is, the current mean-based speedup numbers are flattering and a VisSkip-style gate becomes a *latency-variance* fix rather than a throughput fix.
- Cheap extra read-out in the same run: the per-category MAT table tells you whether GLANCE-2B's "weak on long free-form text" is a long-output effect (MTC/CR at 285–748 tokens) or a grounding effect (ChartQA/CQA) — MMSpec's data says those are different failures (ViSpec is bad at both; MSD is good at both; SAM is catastrophic only on the visual bin).

**Experiment E (only if D's gate looks live, still <1 day, no training).** Implement ViSkip exactly (Eqs. 1–4, τ = 0.35) on top of the frozen GLANCE-2B head: if `S_t > τ`, skip drafting and emit one target token. Measure walltime speedup and MAT with and without the gate. **Step 0 before anything else, 30 minutes:** confirm the stack can actually expose per-step attention over image-token positions in the same forward that produces the verify logits (eager attention or an attention hook). If it cannot, ViSkip is not implementable in this system and that is the finding — check on ~20 prompts before scaling.
- **Falsifies ViSkip for GLANCE if** the gate does not raise decode-only walltime speedup — which is a live risk here, because GLANCE's economics are inverted relative to EAGLE's: on the Qwen3-VL-8B measurement the one-pass draft is only **0.19× an AR forward** while the packed verify is **1.07×** (Table 11; the 2B sibling's ratios will differ but the verify-dominated shape should hold), so replacing a draft+verify round with a plain AR step saves only the ~1.26×-vs-1.00× difference, whereas a mis-timed gate can cost more in lost accepted tokens than it saves. ViSkip's own paper reports no numbers, so this is an open question, not a forecast.

**Cannot be tested cheaply:** anything requiring the ten-method unified harness on four A100s, or a cross-target replication of Table 3 — that was a multi-week artifact and is exactly the kind of thing to reuse rather than rebuild.

## 7. Confidence

**High** that the *protocol and diagnostics* transfer; **low** that ViSkip-the-method transfers.

- Protocol: MAT + walltime + latency CDF + the high-visual/low-visual acceptance probe require no new modules, no training, and no assumptions about the drafter family — they are measurements. They apply to a block-diffusion head unchanged (accepted length per round, per-request wall time, attention-mass split), and they would give GLANCE-2B its first third-party-comparable numbers. Confidence high.
- Diagnostics: the direction of the finding (acceptance degrades on visually grounded steps) is consistent across three very different drafters (MSD, ViSpec, SAM) and both target models, so the *phenomenon* is likely real for GLANCE-2B too. But the magnitudes come from Qwen2.5-VL-7B at batch 1 with 30-token trees and a bonus-excluded MAT, and GLANCE-2B's verify economics are structurally different (one-pass draft at 0.19× vs verify at 1.07×). Confidence medium that the magnitude survives the change of target, drafter family and batch regime.
- ViSkip: an unmeasured, gated fallback to full AR whose benefit depends on per-round cost ratios that GLANCE inverts. **Zero reported effect sizes**, an absent τ sweep, an unspecified K, an unspecified attention layer/head reduction, and a `CrossAttn(h_t, V)` formulation that does not match how either evaluated target represents images — it is not reimplementable from the paper alone. It can only lower accepted length by construction. Confidence low.
- The motivating evidence is weaker than the abstract implies: for the *best* method (MSD) the high-visual penalty is 2.27 → 2.12, i.e. **6.6% relative**; only the already-broken SAM collapses (0.44 → 0.14). So even if the phenomenon replicates on GLANCE-2B, the expected gate gain is single-digit percent — and MMSpec never quantifies it.
- One contradiction that is worth more than the method: MMSpec's step-level story (acceptance dies on visually grounded steps) points one way, while GLANCE's task-level result points the other (the grounded transcription tasks are its *most* draftable: DocVQA τ 3.78, ChartQA 5.16, versus captioning 3.04). GLANCE's drafter reads fused vision-language states rather than raw visual tokens, which is a plausible reason the MMSpec effect might not bind. Resolving that disagreement on GLANCE-2B is Experiment D below.
- One caveat that applies to the whole block: only **15 pages, 10 methods, and no accuracy metrics** — the benchmark is deliberately efficiency-only, so it cannot validate GLANCE's lossless claim. It should be used alongside, not instead of, the fp32 audit GLANCE-2B already passes. And because 2 overall rows + 7 cells of its speedup column are arithmetically impossible, only its MAT/τ column and its diagnostic design should be imported.
