set -uo pipefail
cd /home/elv-grace/model-detection
INPUT=$(python3 -c "
import json
gt=json.load(open('eval/box_gt/box_labels.json'))['frames']
print('\n'.join(f'/elv/test/{n}.png' for n in sorted(n for n,f in gt.items() if f.get('done'))))")
run () {
  name=$1; params=$2
  dest="eval/experiments/14_yoloe_imgsz/runs/$name"
  rm -rf "$dest"; mkdir -p "$dest"
  s=$(date +%s)
  echo "$INPUT" | podman run --rm -i \
    --volume="$(pwd)/eval/frameset/frames:/elv/test:ro" \
    --volume="$(pwd)/$dest:/elv/tags:U" \
    --volume=detection_cache:/root/.cache --network host \
    --device nvidia.com/gpu=3 model-frame-vector \
    --output-path /elv/tags/out.jsonl --params "$params" > "$dest/run.log" 2>&1
  ex=$?; e=$(date +%s)
  echo "$params" > "$dest/params.json"; echo $((e-s)) > "$dest/wall_seconds"
  echo "$name exit=$ex wall=$((e-s))s tags=$(grep -c '\"type\": \"tag\"' "$dest/out.jsonl" 2>/dev/null || echo 0)"
  [ $ex -ne 0 ] && tail -12 "$dest/run.log"
  python3 - "$dest/out.jsonl" <<'PY'
import json, sys, os
p=sys.argv[1]
if not os.path.exists(p): raise SystemExit
lines=open(p).readlines()
with open(p,"w") as h:
    for l in lines:
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        if r.get("type")=="tag": r["data"].pop("vector",None)
        h.write(json.dumps(r)+"\n")
PY
}
B='{"detect_target":["brand"],"brand_detector":"fast"'
run Y1280 "$B}"
run Y1600 "$B,\"brand_imgsz\":1600}"
run Y1920 "$B,\"brand_imgsz\":1920}"
run Y2560 "$B,\"brand_imgsz\":2560}"
run Y1280_2x2 "$B,\"brand_tiles\":\"2x2\"}"
run Y1280_3x2 "$B,\"brand_tiles\":\"3x2\"}"
run Y1280_4x3 "$B,\"brand_tiles\":\"4x3\"}"
run Y2560_2x2 "$B,\"brand_imgsz\":2560,\"brand_tiles\":\"2x2\"}"
