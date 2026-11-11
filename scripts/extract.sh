#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

case "$1" in
  home)    DIR=amazon_home;    CATEGORY="Amazon Home" ;;
  fashion) DIR=amazon_fashion; CATEGORY="AMAZON FASHION" ;;
  beauty)  DIR=all_beauty;     CATEGORY="All Beauty" ;;
  *) echo "Unknown domain: $1 (home, fashion, or beauty)" >&2; exit 1 ;;
esac

for SPLIT in train valid test; do
  python -m extraction.extract_preferences --input "data/$DIR/$SPLIT.jsonl" --output "data/$DIR/${SPLIT}_preferences.json"
done
python -m extraction.extract_attributes --main_category "$CATEGORY" --output "data/item/$DIR/attributes.json"
