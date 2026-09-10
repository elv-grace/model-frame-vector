set -uo pipefail
cd /home/elv-grace/model-detection
run () {
  label=$1; name=$2; params=$3
  dest="eval/experiments/15_temporal/$label/$name"
  [ -f "$dest/wall_seconds" ] && { echo "$label/$name already done"; return; }
  rm -rf "$dest"; mkdir -p "$dest"
  INPUT=$(ls test-files/seg/$label/*.mp4 | sed "s|^test-files/|/elv/test/|")
  s=$(date +%s)
  echo "$INPUT" | podman run --rm -i \
    --volume="$(pwd)/test-files:/elv/test:ro" \
    --volume="$(pwd)/$dest:/elv/tags:U" \
    --volume=detection_cache:/root/.cache --network host \
    --device nvidia.com/gpu=3 model-frame-vector \
    --output-path /elv/tags/out.jsonl --params "$params" > "$dest/run.log" 2>&1
  ex=$?; e=$(date +%s)
  echo "$params" > "$dest/params.json"; echo $((e-s)) > "$dest/wall_seconds"
  echo "$label/$name exit=$ex wall=$((e-s))s tags=$(grep -c '\"type\": \"tag\"' "$dest/out.jsonl" 2>/dev/null || echo 0)"
  [ $ex -ne 0 ] && tail -12 "$dest/run.log"
}
# Both paths, each at its own best tiling from step 2 (coverage 2x2, fast 3x2), plus each
# untiled so the step-3 gain can be separated from the step-2 gain.
for L in nba mitchells; do
  run $L C_cov_tiled  "{\"fps\":2,\"detect_target\":[\"brand\"],\"brand_tiles\":\"2x2\"}"
  run $L C_cov_plain  "{\"fps\":2,\"detect_target\":[\"brand\"]}"
  run $L F_fast_tiled "{\"fps\":2,\"detect_target\":[\"brand\"],\"brand_detector\":\"fast\",\"brand_tiles\":\"3x2\"}"
  run $L F_fast_plain "{\"fps\":2,\"detect_target\":[\"brand\"],\"brand_detector\":\"fast\"}"
done
