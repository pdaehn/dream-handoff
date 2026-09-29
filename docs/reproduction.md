# Reproduction

The immutable repository snapshot for this publication is the revision associated with the GitHub release tag for version `1.0.0`, prefixed with `v`. Run the canonical publication command from that checkout with Python 3.12, `uv`, and the pinned LeRobot revision `c841a0c25833b866970c5a31d66c2bff60f171de`:

```bash
uv sync --locked --dev
```

The publication script defaults to `--device cuda`; use a CUDA-capable host for the pinned result regeneration. It reads external artifacts and writes five JSON results and, when requested, four PNG and four PDF figures. No robot is needed. The [method reference](method.md) defines the analyses; this page specifies their inputs and commands.

## Download the pinned artifacts

Download the four pinned Hugging Face snapshots into a fresh external root. Each repo ID is a current locator, each revision is an immutable repository snapshot, and the SHA-256 values below identify frozen scientific content. The present publication locators and revisions were assigned after physical collection.

```bash
ARTIFACTS=/path/to/dreamhandoff-artifacts
DATASET_ROOT="$ARTIFACTS/dataset/dreamhandoff-so101-rectangle-on-peg"
POLICY="$ARTIFACTS/policy/dreamhandoff-smolvla-rectangle-on-peg"
R2_BUNDLE="$ARTIFACTS/r2/dreamhandoff-r2-rectangle-dynamics"
R2="$R2_BUNDLE/checkpoints/latest.pt"
EVIDENCE="$ARTIFACTS/evidence/prospective-evaluation"
CAPTURE="$EVIDENCE/capture.npz"
GENERATED="$EVIDENCE/generated_banks.npz"

uv run hf download pdaehn/dreamhandoff-so101-rectangle-on-peg \
  --repo-type dataset --revision 70d08de50664a8c973375484c56ca553815719e5 \
  --local-dir "$DATASET_ROOT"
uv run hf download pdaehn/dreamhandoff-smolvla-rectangle-on-peg \
  --revision 7e3d6e4c8ec1d43673b0ca96035e8995ed179601 \
  --local-dir "$POLICY"
uv run hf download pdaehn/dreamhandoff-r2-rectangle-dynamics \
  --revision b3510497b0ff135220ea92468d2efbd5908e4d41 \
  --local-dir "$R2_BUNDLE"
uv run hf download pdaehn/dreamhandoff-so101-prospective-evaluation \
  --repo-type dataset --revision 4ff19fae23719daa0899c04166cb4d445fde0261 \
  --local-dir "$EVIDENCE"
```

The prospective physical evaluation evidence repository contains `capture.npz`, `dataset/`, and `generated_banks.npz`. The [artifact manifest](../configs/artifacts.json) is the machine-readable locator map. Verify the four file identities:

```text
SmolVLA model.safetensors  668cbfa273b30bd023ebcdc246e5fd0d533633bdf0d0d3c6ccd2891aef5dd64c
R2 latest.pt              5260b7c5df88929e21bd60c8686fe7b9dc43db7b2c6238cf49524b42a33ac9fc
Capture capture.npz        c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba
Generated banks           3ce2ab0c8eb39b9650de98f3377f1b676e5131205f808dca16490c73982f3258
```

When the released SmolVLA policy is first instantiated, pinned LeRobot also loads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, the base model named in the policy configuration. That upstream revision was not recorded, so this dependency may be downloaded from its current default revision. The released policy weights themselves are identified by the pinned policy snapshot and `model.safetensors` hash above; [Method details](method.md#dataset-and-model-training-interfaces) records the provenance limitation.

The R2 snapshot also contains `config.json`, byte-identical to the portable [R2 configuration](../configs/r2/rectangle_s12_r64.json). The five canonical result JSONs contain the pinned release revisions. The current calibration JSON SHA-256 is recorded in the [artifact manifest](../configs/artifacts.json).

The R2 loader verifies all eight dataset scientific payload files by SHA-256 before loading and rejects unclassified files. Dataset `README.md` is presentation metadata and `.gitattributes` is repository-management metadata; neither enters scientific payload identity. The [episode split](../configs/data/episode_split.json) uses the current dataset locator. Its episode-metadata SHA-256 is `7ef860f074a7687405a985a1d1cd7ccc04df01c1e6f50dcbcaaab4309a117555`, and its locator-independent scientific split SHA-256 is `e7d93ce2f0c5f7ae60f1dfcdfb2cbce68ff194e1d559e90145f61b97795f6f5a`. The original full-manifest canonical digest `cd0893cdbfd2d65198f8de8e55c8505b89bf40f998c3f846d50d4a42a03ffa21` remains in the artifact manifest as provenance.

## Regenerate the published outputs

```bash
uv run python scripts/reproduce.py \
  --diagnostic "$CAPTURE" \
  --calibration-dataset "$DATASET_ROOT" \
  --calibration-policy "$POLICY" \
  --r2-checkpoint "$R2" \
  --generated-banks "$GENERATED" \
  --output-dir results \
  --figures figures
```

The command verifies input hashes and the deterministic publication calibration JSON. It writes `hysteresis_calibration.json`, `rollout_characterization.json`, `hysteresis.json`, `handoff_prediction.json`, and `prefix_conditioning.json`, plus four PNG and four PDF figures. The four physical analyses use the canonical mapping of 24 saved episodes: 12,257 control rows, 454 activated banks, and 430 non-initial handoffs. The append-only capture contains one discarded re-record excluded by that mapping. The frozen threshold is `hysteresis_tau=0.06851652264595032`.

To compare regeneration against the checked-in files without writing into the checkout, use the same inputs with temporary output directories:

```bash
OUT="$(mktemp -d)"
uv run python scripts/reproduce.py \
  --diagnostic "$CAPTURE" \
  --calibration-dataset "$DATASET_ROOT" \
  --calibration-policy "$POLICY" \
  --r2-checkpoint "$R2" \
  --generated-banks "$GENERATED" \
  --output-dir "$OUT/results" \
  --figures "$OUT/figures"
diff -r results "$OUT/results"
diff -r figures "$OUT/figures"
```

Plotting alone reads the checked-in results and does not load the capture:

```bash
uv run python scripts/plot_results.py --results results --output figures
```

## Validate artifact copies

For a direct hash check:

```bash
sha256sum \
  "$POLICY/model.safetensors" \
  "$R2" \
  "$CAPTURE" \
  "$GENERATED"
```

The external-artifact test gates check capture format, artifact compatibility, and replay payloads without connecting to a robot:

```bash
DREAMHANDOFF_DIAGNOSTIC_ARTIFACT="$CAPTURE" \
DREAMHANDOFF_R2_CHECKPOINT="$R2" \
DREAMHANDOFF_EXP2_GENERATED_BANKS="$GENERATED" \
DREAMHANDOFF_POLICY="$POLICY" \
  uv run pytest tests/test_capture_schema.py tests/test_analysis.py \
    -m 'diagnostic_artifact or inference_artifacts or prefix_policy_replay'
```

The dataset split can be checked against the pinned dataset revision:

```bash
uv run python scripts/generate_split.py \
  --dataset "$DATASET_ROOT" --check configs/data/episode_split.json
```

This checks ordered 96/24 episode membership, the locator-independent scientific digest, and the task/length fingerprint. For direct model artifact gates:

```bash
DREAMHANDOFF_DATASET="$DATASET_ROOT" \
DREAMHANDOFF_R2_CHECKPOINT="$R2" \
DREAMHANDOFF_POLICY="$POLICY" \
  uv run pytest -m 'dataset_artifact or r2_checkpoint or smolvla_artifact'
```

The [runtime configuration](../configs/prospective_evaluation.json), [collection provenance](../configs/prospective_evaluation_provenance.json), and [collection entry point](../scripts/rollout.py) record how the saved physical evaluation was collected. Regeneration uses the pinned saved evidence; it does not recollect episodes or retrain models.
