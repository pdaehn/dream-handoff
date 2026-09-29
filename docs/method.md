# Method and implementation details

This reference defines the released controller and analyses. [README](../README.md) gives the scientific account; [Reproduction](reproduction.md) gives pinned artifacts and commands.

## DreamChunk mechanism

[DreamChunk (DREAM-Chunk), introduced by Chen et al.](https://arxiv.org/abs/2606.18589), supplies the inherited candidate-selection pattern: draw several policy action chunks, imagine their outcomes with a learned world model, and select actions by phase-aligned latent matching. The following implementation details do not make that pattern a DreamHandoff contribution.

### Candidate generation and action spaces

A **candidate** is one policy-generated action chunk; a **candidate bank** is the ten candidates generated from one prepared observation. The worker repeats the observation across ten batch entries, draws independent SmolVLA flow noise, predicts 50-action chunks in one batch, and postprocesses them. It retains policy-space actions for RTC prefix conditioning and canonical six-dimensional control-space actions for R2Dreamer imagination and robot dispatch. The worker does not advance the live world-model posterior.

### World-model scoring and candidate selection

R2Dreamer maintains one recurrent posterior from live observations and previously served canonical actions. When a bank activates, the runtime copies that posterior and imagines each candidate. Its matching feature concatenates flattened categorical stochastic state and deterministic state. For bank phase `p`, the selector compares the live posterior with each candidate's dreamed state *before* action `p` using L2 distance. The closest candidate is the **nearest** candidate; a tie in `torch.argmin` goes to the lowest index.

At phase 0, all candidates share the activation posterior. One draw from a rollout-wide CPU `torch.Generator` seeded once with `0` chooses the initial candidate. Episode reset does not rewind this stream. Later, memoryless selection chooses the nearest candidate on each tick. DreamHandoff's absolute-hysteresis variant is defined below.

### Asynchronous execution in DreamChunk

DreamChunk also describes execution under asynchronous policy inference. While the current candidate bank continues executing, a new policy request is issued and the latent state at that request is stored. When the newly generated bank replaces the current bank, DreamChunk initializes its dreamed trajectories from that stored request-time latent state. DreamHandoff preserves asynchronous candidate generation but changes the grounding semantics at bank activation, as described below.

### Forecasting the reactive selector

During normal execution at time `t`, DreamChunk compares the observation-conditioned live posterior `s_t` with the phase-aligned dreamed state of each candidate. While policy inference runs asynchronously, the robot continues acting, but the observation-conditioned posterior at the future bank activation is not yet available. Predicting exactly which outgoing candidate will be selected then would require forecasting the selector's intervening switches.

A tempting forecast rolls that selector forward in the world model, replacing the unavailable future live posterior with a predicted state. The comparison changes from **observed state ↔ predicted candidate states** to **predicted state ↔ predicted candidate states**. If the predicted state follows candidate `i`'s selected actions, both it and candidate `i`'s dream use the same model dynamics and action sequence. In the deterministic idealization, `ŝ[t+1] = f(ŝ[t], a_i[t])` and `s_i[t+1] = f(s_i[t], a_i[t])`: coincident states stay coincident, and nearby states can remain close. This creates a structural self-consistency or candidate-locking tendency because the real-observation residual that can drive a reactive switch is absent. It is a concern about forecasting across missing future observations, not a claim that every model rollout locks or a failure experimentally proved here.

## DreamHandoff adaptation

### Activation-time re-grounding

LeRobot prepares observations, runs SmolVLA inference and action processing, and dispatches actions. The first bank is generated synchronously. Later requests snapshot a prepared observation and the current served-action count while the active bank continues executing; at most one replacement request is pending or in flight.

DreamHandoff differs from the asynchronous DreamChunk formulation in how the returned bank is grounded. If a request is issued at action count `request_action_count` and activated at `activation_action_count`, the **actual elapsed delay** is

`activation_action_count − request_action_count`.

The controller removes that many leading actions from every returned candidate and caps the usable remainder at 35 actions. Rather than initializing the new dreamed trajectories from the latent state stored when the policy request was issued, it imagines the surviving candidate suffixes from R2Dreamer’s **activation-time posterior**, after the intervening observations and executed actions have updated the live recurrent state.

The bank's origin is the activation action count, its phase starts at zero, and the bank-local selector incumbent is cleared. A result with no usable actions, or an active bank that expires before replacement, raises an error. A policy request and a bank **handoff** are therefore distinct events: the request begins candidate generation, while the handoff occurs when the completed bank activates and replaces the current one.

### Switching and hysteresis

A **switch** changes selected candidate within one active bank. An **A→B→A reversal** is a return to candidate A through B on consecutive control rows. A **rapid successive switch** occurs within three control rows of the previous switch. Hysteresis keeps the incumbent unless

```text
distance(incumbent) − distance(nearest) > hysteresis_tau
```

The inequality is strict. The released `hysteresis_tau` is `0.06851652264595032`. The incumbent resets at each bank activation, while the phase-0 RNG stream continues. This rule controls within-bank switches; a new bank is a handoff, not a switch between indices in the old bank.

### Prefix-conditioned handoffs

The optional `static_rtc` mode takes each outgoing candidate's next 15 policy-space actions, starting at the request phase, and assigns that same-index prefix to the corresponding new candidate. LeRobot's RTC processor receives a predicted `inference_delay` from the maximum observed policy latency converted to control ticks (ceiling) and an `execution_horizon` of 15. The initial bank remains unguided; a warmup generation supplies the first latency estimate. The offline **Static-live** reconstruction uses the recorded prediction and taper semantics. The prospective physical capture used `plain` generation; Static-live is not a separately collected robot condition.

## R2Dreamer integration

### Source and scope

DreamHandoff vendors an adapted subset of [NM512/r2dreamer](https://github.com/NM512/r2dreamer) at commit `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`, under its MIT license and Copyright (c) 2026 Naoki Morihira. The full notice is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). `DreamHandoffInferenceEngine` uses `R2DreamerRuntime` directly.

The adapted subset retains image and vector encoders, a block-GRU recurrent state-space model (RSSM) with categorical posterior and dynamics prior, deterministic action-conditioned imagination, and feature construction. Projector weights remain in the strict checkpoint state, although live inference does not call the projector. `imagine_final_state` serves the handoff-prediction analysis outside live control.

DreamHandoff-specific preprocessing, action normalization, checkpoint loading, and runtime code provide the live API. The focused trainer uses dynamics KL, representation KL, and Barlow Twins losses, without a decoder or control objective.

### Checkpoint schema and compatibility

The final artifact is `dreamhandoff-r2-rectangle-dynamics/checkpoints/latest.pt`, SHA-256 `5260b7c5df88929e21bd60c8686fe7b9dc43db7b2c6238cf49524b42a33ac9fc`. Its checkpoint format is `dream-reflex-r2dreamer`, version 2, at training step 25,000, with architecture signature `560ed2c44b18e3fcf573160ca9ace32f26390d69797c9ea5c62f0b7cdb82e5e2`. The `dream-reflex` string is a frozen schema identifier kept for compatibility.

The loader rejects incompatible schema, revision, dimensions, action metadata, architecture fingerprint, or model parameters. Training-only optimizer and scheduler state are not loaded at runtime. `latest.pt` is the frozen final-result parity target.

### Inputs, preprocessing, and runtime state

The checkpoint requires `observation.images.context` followed by `observation.images.wrist`. Each image is scaled to `[0,1]`, bilinearly resized to 64×64, and concatenated into six channels. The unscaled six-dimensional `observation.state` is encoded alongside them.

The checkpoint stores six ordered canonical action names and per-dimension q01/q99 values. Before entering the RSSM, each action is transformed by `2 × (action − q01) / (q99 − q01) − 1`, without clipping; a zero quantile range uses checkpoint epsilon `1e-8`. The R2 transition retains upstream tail saturation. R2 consumes canonical actions and does not denormalize controller output.

`observe(observation, previous_action)` maintains one recurrent, observation-conditioned posterior. The first observation after construction or reset begins from zero stochastic and deterministic states and masks the previous action. Later observations transition from the prior posterior using the action executed between observations. Candidate imagination copies and expands the activation posterior, leaving the live posterior unchanged. Categorical argmax modes make imagination deterministic.

For actions shaped `[N,H,A]`, native R2 yields post-action states. The runtime prepends the shared activation posterior and drops the last post-action state to expose exactly `H` pre-action selector phases. Matching flattens the categorical state in stored `[stoch, discrete]` order and appends the deterministic state: `32 × 16 + 2048 = 2560` float features. A fixed candidate/horizon shape may be compiled on CUDA; CPU and unmatched shapes use eager execution.

### Dataset and model training interfaces

The focused R2 training entry point is [`scripts/train_r2.py`](../scripts/train_r2.py).

The [episode split](../configs/data/episode_split.json) fixes 96 training and 24 validation episodes from the 120-episode dataset. The loader verifies dataset identity and scientific payload hashes; it never generates a split. [Reproduction](reproduction.md) records the split digests and pinned LeRobot revision.

The deployed SmolVLA policy was fine-tuned with upstream LeRobot on the 96 training episodes; its chunk size and action-step setting are both 50. The [policy configuration](../configs/smolvla/rectangle.json) records training settings, camera mapping, and action order. The pretrained source was `lerobot/smolvla_base`; no immutable pretrained revision was recorded. A newly trained stochastic run is not claimed to recreate the released weights byte for byte.

The focused [R2 configuration](../configs/r2/rectangle_s12_r64.json) uses 64-step episode-local windows, batch size 16, and 25,000 optimizer updates. Samples include preceding actions; `is_first` resets the RSSM at episode starts, and windows do not cross episodes. Training and validation sampling, including deterministic validation modes, are defined by the linked configuration and [`scripts/train_r2.py`](../scripts/train_r2.py).

Optional R2 retraining decodes the dataset videos. LeRobot's default `torchcodec` backend requires compatible system FFmpeg shared libraries; on a host without them, pass `--video-backend pyav` to `scripts/train_r2.py`. The locked environment includes PyAV. Retraining materializes a multi-gigabyte episode cache and is not part of the released-evidence reproduction command.

Training uses straight-through Gumbel categorical posterior samples, dynamics and representation KL losses, and Barlow Twins between the projected RSSM feature and encoder embedding. The [R2 configuration](../configs/r2/rectangle_s12_r64.json) and trainer specify their scales, optimizer, scheduler, and gradient handling.

`latest.pt` contains the model, preprocessing and action metadata, training step, and training state. The released checkpoint is the inference and result-parity target; [Reproduction](reproduction.md) gives artifact checks and commands.

## Evaluation methodology

### Evidence and evaluation setting

The physical evidence is 24 saved episodes and 12,257 control rows. A canonical capture-to-dataset mapping excludes one discarded append-only re-record. It contains 454 activated banks and 430 non-initial handoffs. Full-precision outputs are in five result JSON files: [rollout_characterization.json](../results/rollout_characterization.json), [hysteresis.json](../results/hysteresis.json), [hysteresis_calibration.json](../results/hysteresis_calibration.json), [handoff_prediction.json](../results/handoff_prediction.json), and [prefix_conditioning.json](../results/prefix_conditioning.json). The four prospective physical analyses use this same mapping.

### Candidate ranking and score geometry

The **recorded-continuation proxy** at bank phase `p` is the L2 distance between a candidate's canonical action at `p` and the recorded `observation.state` at `bank_origin + p + 1`. The **oracle candidate** minimizes that error among the ten candidates on a row. **Top-1 agreement** is the share of valid rows on which the nearest-by-R2 candidate is this proxy oracle; the released offline matrix gives 18.255% over 11,854 valid rows (20.0% calibration, 17.4% held-out). **Regret** is the chosen candidate's proxy error minus the oracle's proxy error, averaged over valid rows. This oracle uses a factual recorded continuation and does not estimate the unobserved outcome of executing another candidate. The released analyses report top-1 agreement; top-k, Spearman, and pairwise ranking accuracy are not published metrics here. The ranking signal is measurable but limited.

On operational rows (`phase > 0`), sort candidate L2 matching distances independently: `d1` is nearest distance, and `d2 − d1` is runner-up separation. The first asks how well the bank covers the live posterior, as in DreamChunk; the second asks whether DreamHandoff's nearest candidate is decisively better than the runner-up. Over 11,803 rows, `d1` has mean 1.97, median 1.96, p90 3.36; separation has mean 0.090, median 0.035, p90 0.236. These are learned-feature distances, not task errors. Small separations on many rows are consistent with, but do not by themselves prove, switching under repeated argmin selection.

### Hysteresis and within-bank switching

Hysteresis tests how much within-bank stability can be gained when a weakly separated nearest candidate is reconsidered each tick, and what recorded-continuation proxy cost accompanies it.

The fixed calibration uses bank origins `range(0, episode_length, 35)`, ten 50-action candidates, a 35-phase legal bank horizon, and a rollout-wide CUDA Torch noise stream seeded once with `0`. Its runtime-faithful selector uses the phase-0 CPU Torch stream described above. Eight episodes form the calibration split and 16 the held-out split. A 45-threshold sweep consists of 40 linear points from zero to the calibration advantage q90, plus q95, q97.5, q99, q99.5, and q99.9. Points are Pareto filtered by switch rate and mean proxy regret. The chosen point has the lowest switch rate among Pareto points whose mean regret is at most 5% above memoryless selection: `hysteresis_tau=0.06851652264595032`.

On calibration rows, switch rate changes from 23.61% to 13.51% and mean proxy regret from 0.855 to 0.895. Held-out switch rate changes from 23.72% to 12.89%; observed regret delta (hysteresis minus memoryless) is 0.0502. A paired bank-clustered bootstrap over 228 `(episode, bank_origin)` units uses 2,000 resamples, seed `0`, and a 2.5/97.5 percentile interval. Its bootstrap mean regret delta is 0.0501 and 95% interval is [0.0297, 0.0719]. The observed delta and bootstrap mean are distinct quantities. This offline proxy regret is distinct from the prospective matching-distance penalty.

Selector replay compares memoryless nearest selection with the frozen hysteresis rule on 11,350 operational within-bank transitions. Memoryless has 1,980 switches (17.44%), 1,050 rapid successive switches, and 163 A→B→A reversals; hysteresis has 846 (7.45%), 257, and 33. Hysteresis's mean latent matching-distance penalty is 0.0050. The phase-0-to-1 transition has separate initialization semantics and is excluded from these operational switch rates.

### Handoff characterization

#### Boundary discontinuity

First and second canonical action-space finite differences are measured at bank boundaries and strict bank interiors, excluding episode boundaries. The strict interior uses a 10-action margin, phases 10 through 39 of the 50-action bank representation. Boundary versus strict-interior means are 4.13 versus 1.39 for first difference and 4.65 versus 1.36 for second difference; p95 values are 10.86 versus 3.45 and 12.93 versus 2.94. They measure action commands, not physical velocity or acceleration. The [rollout diagnostic figure](../figures/rollout_discontinuity.png) shows the distributions.

#### Activation-time grounding

This diagnostic asks how faithfully open-loop world-model propagation over the request-to-activation interval can recover the posterior later obtained from real observations. It supplies empirical context for the forecasting concern above; it does not test or prove candidate locking.

For each non-initial handoff, the analysis compares the imagined endpoint from the **exact factual action history** between request and activation with the actual activation posterior. A **target-selected best-of-ten** comparator retrospectively selects the static candidate endpoint closest to that posterior; an **action-path-nearest** comparator selects the static request-to-activation path minimizing `sqrt(mean_time(sum_action_dim((static − factual)^2)))` before endpoint comparison. The signed difference is `static error − exact-history error`; positive favors exact history. Intervals use 5,000 episode-clustered bootstrap resamples with seed `0`.

| Selector switches before activation | Events | Factual path mismatch | Target-selected difference [95% CI] | Action-path-nearest difference [95% CI] |
|---|---:|---:|---:|---:|
| 0 | 255 | 0 | −0.0243 [−0.0367, −0.0142] | 0 [0, 0] |
| 1 | 119 | 0.764 | −0.0151 [−0.0367, 0.0070] | 0.0380 [0.0154, 0.0651] |
| ≥2 | 56 | 1.157 | −0.0268 [−0.0566, 0.0046] | 0.0558 [0.0237, 0.0885] |
| All | 430 | 0.362 | −0.0221 [−0.0342, −0.0110] | 0.0178 [0.0090, 0.0285] |

Overall mean endpoint error is 1.77 for target-selected static, 1.81 for action-path-nearest static, and 1.80 for exact history. As selector switches accumulate, the factual action path increasingly differs from any single static candidate. For zero-switch events, the factual path is itself a static candidate, so target-selected best-of-ten is structurally no worse than that candidate apart from numerical effects. Even propagation under the factual executed actions is not closest under every comparator: the hindsight target-selected static endpoint can be closer in latent space, while action-path-nearest static gives the opposite ordering. Thus the analysis does not establish one universally superior grounding path, but open-loop endpoint prediction is not a clean substitute for the eventual observation-conditioned posterior. The [grounding diagnostic figure](../figures/handoff_prediction.png) shows the switch strata.

### Prefix-conditioned handoffs

Prefix conditioning tests continuity across banks without forecasting the exact future outgoing switch sequence. The conditions distinguish an unguided baseline, a coherent same-index outgoing trajectory, a hindsight factual trajectory, and a deployment-style schedule.

**Command seam** is the L2 discontinuity between action commands across a handoff; **action-difference seam** is the L2 discontinuity in their local action changes. Both are measured in canonical action space. Plain is unguided request-time generation. Static uses the outgoing same-index candidate prefix and GT uses factual actions, each with `L = inference_delay = execution_horizon = L_e`, the realized event delay. Static-live uses a 15-action outgoing prefix, recorded predicted delay, and execution horizon 15. Reactive is an activation-time SmolVLA reference, not an optimal-controller oracle. All 430 events share handoffs, request observations, and paired generation noise; the generated-banks artifact includes these conditions.

Plain has no outgoing-trajectory prefix information. GT asks how much seam improvement factual future actions could provide under matched delay; it is a hindsight diagnostic, not a deployable controller or optimal-policy oracle. Static asks whether one coherent same-index outgoing candidate can recover much of that benefit without predicting the reactive selector's future switches. Static-live represents the implementable recorded predicted-delay and 15-action execution-horizon schedule.

| Condition | Mean command seam | Mean action-difference seam |
|---|---:|---:|
| Plain | 4.10 | 4.79 |
| Static | 2.77 | 3.31 |
| GT | 2.52 | 3.15 |
| Static-live | 2.10 | 2.29 |

Under matched `L_e`, Static captures 84.1% of GT's command-seam benefit and 90.3% of its action-difference-seam benefit relative to Plain: a coherent outgoing prefix recovers most of the hindsight factual-prefix seam benefit without knowing the future switch sequence. Static-live versus Plain is the deployment-schedule comparison. Static-live and GT have different scheduling and taper semantics and cannot be causally ranked by their seam values. Mean candidate diversity remains 0.823–0.877 across the four displayed conditions; reactive-reference distance and state/grounding diagnostics are in [prefix_conditioning.json](../results/prefix_conditioning.json). Static-live's state-seam diagnostic does not support a generic state-continuity claim. Paired seam intervals use 5,000 physical-episode-clustered resamples with seed `0`.

## Interfaces and implementation notes

The concrete controller is [`DreamHandoffInferenceEngine`](../src/dream_handoff/inference/engine.py); selector, bank, sampler, and validated runtime settings are in [`src/dream_handoff/inference/`](../src/dream_handoff/inference/). The R2 runtime is in [`src/dream_handoff/r2dreamer/`](../src/dream_handoff/r2dreamer/). Dataset feature names, R2 action names, and robot action keys must agree; the controller rejects incompatible interfaces rather than silently reordering model actions. The [runtime configuration](../configs/prospective_evaluation.json) and [scientific provenance](../configs/prospective_evaluation_provenance.json) record the frozen physical collection settings. Artifact locators and exact reproduction commands are in [Reproduction](reproduction.md).
