#!/bin/bash
#
# Run the shipped config against the step-1 config over a sampled title.
#
#   ./eval/box_gt/run_title_ab.sh nba              # frames, if sample_frames.sh has run
#   ./eval/box_gt/run_title_ab.sh mitchells seg    # segments, from sample_title.sh
#
# Everything goes into ONE container invocation per config, so the several GB of weights load
# once rather than per input. Output: eval/experiments/11_titles/<label>/<config>/out.jsonl.
#
# `frames` is the default because the question here is presence -- see sample_frames.sh for why
# a wide frame sample beats a few long segments for that, and why `seg` is still the right mode
# when the question is dwell time.

set -uo pipefail
cd "$(dirname "$0")/../.."

LABEL="$1"
MODE="${2:-frames}"
: "${ELV_MODEL_TEST_GPU_TO_USE:=3}"
IMAGE_NAME="${IMAGE_NAME:-model-frame-vector}"
if [ "$MODE" = "frames" ]; then
    SEG="test-files/frames/$LABEL"; GLOB="*.jpg"
else
    SEG="test-files/seg/$LABEL"; GLOB="*.mp4"
fi
OUT_ROOT="eval/experiments/11_titles/$LABEL"

SIX='["logo","letter logo","car logo","emblem","brand","label"]'
FOUR='["logo","letter logo","brand","car logo"]'

declare -A CONFIGS=(
  [A_shipped_fast]="{\"fps\":1,\"class_prompts\":{\"brand\":$SIX},\"brand_detector\":\"fast\"}"
  [B2_step1_conf07]="{\"fps\":1,\"class_prompts\":{\"brand\":$FOUR},\"brand_detector\":\"coverage\",\"brand_conf\":0.07,\"max_detections\":100}"
)

INPUT=$(ls "$SEG"/$GLOB | sed "s|^test-files/|/elv/test/|")
echo "$LABEL ($MODE): $(echo "$INPUT" | wc -l) inputs"

for name in "${!CONFIGS[@]}"; do
    dest="$OUT_ROOT/$name"
    echo "=== $LABEL / $name"
    rm -rf "$dest"; mkdir -p "$dest"
    start=$(date +%s)
    echo "$INPUT" | podman run --rm -i \
        --volume="$(pwd)/test-files:/elv/test:ro" \
        --volume="$(pwd)/$dest:/elv/tags:U" \
        --volume=detection_cache:/root/.cache \
        --network host \
        --device "nvidia.com/gpu=${ELV_MODEL_TEST_GPU_TO_USE}" \
        "${IMAGE_NAME}" \
        --output-path /elv/tags/out.jsonl \
        --params "${CONFIGS[$name]}" > "$dest/run.log" 2>&1
    ex=$?; end=$(date +%s)
    echo "${CONFIGS[$name]}" > "$dest/params.json"
    echo $((end - start)) > "$dest/wall_seconds"
    if [ $ex -ne 0 ]; then
        echo "  !! exit $ex"; tail -20 "$dest/run.log"; continue
    fi
    # Vectors are kept here, unlike the box-GT runs: the title check is a RETRIEVAL question
    # ("is Tissot findable at all"), and that needs the crop embeddings.
    echo "  $(grep -c '"type": "tag"' "$dest/out.jsonl") tags in $((end - start))s"
done
cp "$SEG/segments.json" "$OUT_ROOT/segments.json" 2>/dev/null || true
echo "$MODE" > "$OUT_ROOT/input_mode"
