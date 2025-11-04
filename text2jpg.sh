#!/usr/bin/env bash
set -e

# Check input
if [ $# -lt 1 ]; then
  echo "Usage: ./text2jpg.sh 'Your text here'"
  exit 1
fi

TEXT="$1"
OUTFILE="text_$(date +%Y%m%d_%H%M%S).jpg"

# Image settings
WIDTH=800
HEIGHT=300
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
POINTSIZE=36
FG_COLOR="black"
BG_COLOR="white"

# Ensure ImageMagick is available
if ! command -v convert >/dev/null 2>&1; then
  echo "Error: ImageMagick not found. Install it using:"
  echo "sudo apt install -y imagemagick"
  exit 1
fi

# Generate JPG with text centered
convert -background "$BG_COLOR" -fill "$FG_COLOR" \
  -font "$FONT" -pointsize "$POINTSIZE" \
  -size "${WIDTH}x${HEIGHT}" -gravity center \
  caption:"$TEXT" "$OUTFILE"

echo "Image saved as $OUTFILE"
