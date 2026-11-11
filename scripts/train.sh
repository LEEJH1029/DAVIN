#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

case "$1" in
  home)    DIR=amazon_home;    CATEGORY="Amazon Home" ;;
  fashion) DIR=amazon_fashion; CATEGORY="AMAZON FASHION" ;;
  beauty)  DIR=all_beauty;     CATEGORY="All Beauty" ;;
  *) echo "Unknown domain: $1 (home, fashion, or beauty)" >&2; exit 1 ;;
esac

python train.py --main_category "$CATEGORY" \
  --train_path "data/$DIR/train_preferences.json" --valid_path "data/$DIR/valid_preferences.json" \
  --test_path "data/$DIR/test_preferences.json" --item_attr_path "data/item/$DIR/attributes.json" \
  --output_dir "runs/$DIR" --cache_dir "cache/$DIR"
