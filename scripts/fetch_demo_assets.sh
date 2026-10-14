#!/bin/sh
# Download the demo's third-party runtime files into site/vendor/ and check each
# against the SHA-256 pinned in scripts/demo_assets.txt. Any mismatch aborts, so a
# tampered CDN file is never published. Used by the Pages workflow and for local runs.
set -eu
cd "$(dirname "$0")/.."
grep -v '^#' scripts/demo_assets.txt | while read -r sha path url; do
    dest="site/vendor/$path"
    mkdir -p "$(dirname "$dest")"
    curl -fsSL "$url" -o "$dest"
    actual=$(sha256sum "$dest" 2>/dev/null || shasum -a 256 "$dest")
    if [ "${actual%% *}" != "$sha" ]; then
        echo "checksum mismatch for $path" >&2
        exit 1
    fi
done
echo "demo assets ready in site/vendor/"
