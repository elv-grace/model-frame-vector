#!/bin/bash
#
# Extract single frames spread evenly across the WHOLE runtime of a title.
#
#   ./eval/box_gt/sample_frames.sh test-files/val/basketball_allstars.mp4 nba 15
#
# This is the sampler to use for a PRESENCE question -- "does brand X appear anywhere" -- and
# `sample_title.sh` is the one to use for a DURATION question.
#
# The difference matters more than it sounds. Segment sampling buys temporal continuity, which
# is what tracking and dwell-time need, but it spends the whole budget on a few places in the
# runtime: twelve 45s segments of a 4h16m game is 3.5% of it, and the 45 frames inside one
# segment share a camera, a possession and a set of hoardings, so they are nearer one look than
# forty-five. Brands are not evenly distributed -- a courtside board rotates, a sponsor appears
# in one quarter -- so a sparse-but-clustered sample under-counts them badly.
#
# One frame every N seconds across 100% of the runtime costs the same GPU time and sees every
# part of the title. Keyframe seeking makes it cheap (~0.6s/frame even on a 35 GB source), and
# the seeks are independent so they parallelise.
#
# Output: test-files/frames/<label>/<label>_t<seconds>.jpg -- the timecode is in the filename,
# so `source_media` alone locates a detection in the original title.

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC="$1"; LABEL="$2"; EVERY="${3:-15}"; JOBS="${4:-8}"
DEST="test-files/frames/$LABEL"

DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC" | cut -d. -f1)
# Skip the first and last 2%: titles, credits and studio bumpers are wall-to-wall logos and
# would flatter the detector on content nobody reports on.
LEAD=$((DURATION / 50))
LAST=$((DURATION - LEAD))
COUNT=$(( (LAST - LEAD) / EVERY ))

echo "$SRC: ${DURATION}s -> $COUNT frames, one every ${EVERY}s, ${LEAD}s..${LAST}s"
rm -rf "$DEST"; mkdir -p "$DEST"

seq "$LEAD" "$EVERY" "$LAST" | xargs -P "$JOBS" -I{} \
    ffmpeg -nostdin -y -loglevel error -ss {} -i "$SRC" -frames:v 1 -q:v 2 \
           "$DEST/${LABEL}_t{}.jpg"

echo "wrote $(ls "$DEST"/*.jpg | wc -l) frames, $(du -sh "$DEST" | cut -f1)"
