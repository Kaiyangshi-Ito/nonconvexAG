#!/bin/bash
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

check_deps() {
  local missing=()
  command -v jq >/dev/null 2>&1 || missing+=("jq")
  command -v openssl >/dev/null 2>&1 || missing+=("openssl")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing dependencies: ${missing[*]}"
    echo "Install with: sudo apt-get install ${missing[*]}"
    exit 1
  fi
}

verify() {
  local h="$1"
  if [ ! -f "$DIR/$h.txt" ]; then
    echo "Not found: $h"
    return 1
  fi
  echo "=== $h ==="
  [ -f "$DIR/$h.json" ] && echo "Created: $(jq -r '.created_at // "unknown"' "$DIR/$h.json" 2>/dev/null)"
  if [ -f "$DIR/$h.txt.ots" ]; then
    echo ""
    echo "--- OpenTimestamps ---"
    if command -v ots >/dev/null 2>&1; then
      ots verify "$DIR/$h.txt" 2>&1 | head -5 || echo "Pending or verification failed"
    else
      echo "ots not installed (pip install opentimestamps-client)"
    fi
  fi
  for tsr in "$DIR/${h}"_*.tsr; do
    [ -f "$tsr" ] || continue
    name=$(basename "$tsr" .tsr | sed "s/${h}_//")
    echo ""
    echo "--- $name ---"
    if [ "$name" = "freetsa" ] && [ -f "$DIR/freetsa_cacert.pem" ]; then
      openssl ts -verify -in "$tsr" -queryfile "$DIR/$h.tsq" \
        -CAfile "$DIR/freetsa_cacert.pem" -untrusted "$DIR/freetsa_tsa.crt" 2>&1 | head -3
    else
      openssl ts -verify -in "$tsr" -queryfile "$DIR/$h.tsq" 2>&1 | head -3
    fi
  done
}

check_deps
case "${1:-}" in
  "") echo "Usage: $0 <hash|--all|--latest|--pending>" ;;
  --all)
    found=0
    for f in "$DIR"/*.txt; do
      [ -f "$f" ] || continue
      h=$(basename "$f" .txt)
      [[ "$h" =~ ^[a-f0-9]{40}$ ]] || continue
      found=1; verify "$h"; echo ""
    done
    [ "$found" -eq 0 ] && echo "No timestamp proofs found"
    ;;
  --latest)
    latest=$(find "$DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | xargs -r ls -t | head -1)
    if [ -z "$latest" ]; then echo "No timestamp proofs found"; exit 1; fi
    h=$(jq -r '.commit' "$latest" 2>/dev/null)
    if [ -z "$h" ] || [ "$h" = "null" ]; then echo "Invalid manifest: $latest"; exit 1; fi
    verify "$h"
    ;;
  --pending)
    if ! command -v ots >/dev/null 2>&1; then echo "ots not installed"; exit 1; fi
    found=0
    for f in "$DIR"/*.ots; do
      [ -f "$f" ] || continue
      ots info "$f" 2>&1 | grep -qi bitcoin || { echo "$f"; found=1; }
    done
    [ "$found" -eq 0 ] && echo "No pending proofs"
    ;;
  *) verify "$1" ;;
esac
