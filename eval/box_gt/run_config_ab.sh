#!/bin/bash
#
# Run the PRODUCTION tagger (run.py -> FrameVectorModel) over the 25 box-GT frames under
# several --params configs, so a config change can be scored end to end.
#
# This is deliberately not eval/tools/run_detectors.py. That harness runs each backend
# ungated and skips embedding, which is right for ranking detectors and wrong for asking
# "what does THIS config file deliver" -- the answer to that has to include the confidence
# gate, the synonym NMS, max_detections, min_crop_pixels and the crop step, all of which
# only exist on the production path.
#
#   ./eval/box_gt/run_config_ab.sh            # all configs
#   ./eval/box_gt/run_config_ab.sh A B        # named configs only
#
# Output: eval/experiments/10_config_ab/runs/<name>/out.jsonl, scored by score_config.py.

set -uo pipefail

: "${ELV_MODEL_TEST_GPU_TO_USE:=3}"
IMAGE_NAME="${IMAGE_NAME:-model-frame-vector}"

cd "$(dirname "$0")/../.."
OUT_ROOT="eval/experiments/10_config_ab/runs"

SIX='["logo","letter logo","car logo","emblem","brand","label"]'
FOUR='["logo","letter logo","brand","car logo"]'

# name -> --params JSON. brand only: person is served by a different backend and is not
# what is being changed, and dropping it halves the runtime.
declare -A CONFIGS=(
  # A  what ships today: yoloe26 @1280, conf 0.007, six prompts, cap 30.
  [A_shipped_fast]="{\"class_prompts\":{\"brand\":$SIX},\"brand_detector\":\"fast\"}"
  # A2 the other shipped option, unchanged: isolates "switch backend" from "lower the gate".
  [A2_shipped_coverage]="{\"class_prompts\":{\"brand\":$SIX},\"brand_detector\":\"coverage\"}"
  # B  step 1: coverage backend, gate dropped to 0.05, cap raised, dead prompts removed.
  [B_step1_conf05]="{\"class_prompts\":{\"brand\":$FOUR},\"brand_detector\":\"coverage\",\"brand_conf\":0.05,\"max_detections\":100}"
  # B2 the cheaper rung of the same change, for the cost/coverage trade.
  [B2_step1_conf07]="{\"class_prompts\":{\"brand\":$FOUR},\"brand_detector\":\"coverage\",\"brand_conf\":0.07,\"max_detections\":100}"
)

WANTED=("$@")
[ ${#WANTED[@]} -eq 0 ] && WANTED=("${!CONFIGS[@]}")

# The 25 frames with box ground truth, as container-side paths.
INPUT=$(python3 - <<'PY'
import json, os
gt = json.load(open("eval/box_gt/box_labels.json"))["frames"]
for name in sorted(n for n, fr in gt.items() if fr.get("done")):
    print(f"/elv/test/{name}.png")
PY
)
echo "$INPUT" | wc -l | xargs echo "frames:"

for name in "${WANTED[@]}"; do
    params="${CONFIGS[$name]}"
    dest="$OUT_ROOT/$name"
    echo "=== $name :: $params"
    rm -rf "$dest"; mkdir -p "$dest"

    start=$(date +%s.%N)
    echo "$INPUT" | podman run --rm -i \
        --volume="$(pwd)/eval/frameset/frames:/elv/test:ro" \
        --volume="$(pwd)/$dest:/elv/tags:U" \
        --volume=detection_cache:/root/.cache \
        --network host \
        --device "nvidia.com/gpu=${ELV_MODEL_TEST_GPU_TO_USE}" \
        "${IMAGE_NAME}" \
        --output-path /elv/tags/out.jsonl \
        --params "$params" > "$dest/run.log" 2>&1
    ex=$?
    end=$(date +%s.%N)

    echo "$params" > "$dest/params.json"
    printf '%s\n' "$(echo "$end - $start" | bc)" > "$dest/wall_seconds"
    # Drop the vectors before the run is stored. Scoring reads boxes and scores only, and a
    # 768-float array per tag is ~97% of the file -- 320 MB across these runs against 8 MB
    # without. The vectors are reproducible from the params and are not the artifact here.
    python3 - "$dest/out.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as handle:
    lines = handle.readlines()
with open(path, "w") as handle:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") == "tag":
            # Drop the vector-less `output_tags` twin BEFORE stripping vectors. Once the
            # vectors are gone the twin is byte-identical to its parent -- it shares the same
            # additional_info -- and every box downstream would be counted twice.
            if record["data"].get("vector") is None:
                continue
            record["data"].pop("vector", None)
        handle.write(json.dumps(record) + "\n")
PY
    if [ $ex -ne 0 ]; then
        echo "!!! $name exited $ex; tail of log:"; tail -20 "$dest/run.log"
    else
        echo "    $(wc -l < "$dest/out.jsonl" 2>/dev/null || echo 0) tags in $(cat "$dest/wall_seconds")s"
    fi
done
