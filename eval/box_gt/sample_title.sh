#!/bin/bash
#
# Cut a stratified sample of short segments out of a full-length title.
#
# A whole title is not runnable in an evaluation loop: the all-star game is 4h16m of 1080p60
# in 35 GB, and at 1 fps that is 15,000 frames per config. Twelve segments spread evenly across
# the runtime cover the thing that actually varies -- camera, lighting, which hoardings are in
# shot, which act of the film -- at 1/25th the cost.
#
# Segments stay SEPARATE files rather than being concatenated, so `source_media` identifies the
# segment and the original timecode is recoverable as (segment start + frame_idx / fps). A
# concatenated file loses that, and for a brand coverage report the timecode is the product.
#
#   ./eval/box_gt/sample_title.sh test-files/val/basketball_allstars.mp4 nba 12 45
#
# Output: test-files/seg/<label>/<label>_<NNN>_at<seconds>.mp4 plus segments.json.

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC="$1"; LABEL="$2"; COUNT="${3:-12}"; LEN="${4:-45}"
DEST="test-files/seg/$LABEL"

DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC" | cut -d. -f1)
echo "$SRC: ${DURATION}s -> $COUNT x ${LEN}s segments"

rm -rf "$DEST"; mkdir -p "$DEST"

# Skip the first and last 4% -- logos and studio bumpers cluster in titles and credits, and a
# sample dominated by them would flatter the detector on content nobody reports on.
LEAD=$((DURATION * 4 / 100))
SPAN=$((DURATION - 2 * LEAD - LEN))

python3 - "$COUNT" "$LEAD" "$SPAN" > "$DEST/starts.txt" <<'PY'
import sys
count, lead, span = (int(v) for v in sys.argv[1:4])
for i in range(count):
    print(lead + (span * i) // max(1, count - 1))
PY

i=0
while read -r START; do
    OUT=$(printf "%s/%s_%03d_at%s.mp4" "$DEST" "$LABEL" "$i" "$START")
    # -ss BEFORE -i keyframe-seeks, which is what makes this affordable on a 35 GB file.
    # Re-encoded rather than stream-copied: a copy starts at the previous keyframe and the
    # first seconds decode as garbage, which would be scored as real frames.
    # -nostdin, because ffmpeg reads stdin by default and would swallow the rest of the
    # start list this loop is reading from -- producing a few truncated segments and no error.
    ffmpeg -nostdin -y -loglevel error -ss "$START" -t "$LEN" -i "$SRC" \
           -an -vf fps=6 -c:v libx264 -crf 22 "$OUT" 2>/dev/null || {
        echo "  !! segment at ${START}s failed, skipping"; i=$((i+1)); continue; }
    echo "  $(basename "$OUT")  $(du -h "$OUT" | cut -f1)"
    i=$((i+1))
done < "$DEST/starts.txt"

python3 - "$DEST" "$LABEL" "$SRC" "$LEN" > "$DEST/segments.json" <<'PY'
import glob, json, os, re, sys
dest, label, src, length = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = {"source": src, "segment_seconds": length, "decoded_fps": 6, "segments": {}}
for path in sorted(glob.glob(f"{dest}/{label}_*.mp4")):
    start = int(re.search(r"_at(\d+)\.mp4$", path).group(1))
    out["segments"][os.path.splitext(os.path.basename(path))[0]] = start
print(json.dumps(out, indent=1))
PY

echo "wrote $DEST/segments.json ($(ls "$DEST"/*.mp4 | wc -l) segments, $(du -sh "$DEST" | cut -f1))"
