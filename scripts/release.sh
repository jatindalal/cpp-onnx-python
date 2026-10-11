#!/usr/bin/env bash
#
# Build the macOS wheel, test it in a clean venv, and publish every wheel in
# dist/ (this macOS wheel plus the Linux wheels from scripts/linux_wheels.sh)
# as GitHub Release assets. Requires `gh` (https://cli.github.com). Options:
#
#   scripts/release.sh                  build + test + publish tag v<pkg-version>
#   scripts/release.sh --tag v0.1.1     publish under a specific tag
#   scripts/release.sh --no-release     build + test only (no network)
#   scripts/release.sh --no-test        skip the clean-venv test
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPO_SLUG="jatindalal/cpp-onnx-python"
TAG=""
DO_TEST=1
DO_RELEASE=1

while [[ $# -gt 0 ]]; do
	case "$1" in
		--tag) TAG="$2"; shift 2 ;;
		--no-test) DO_TEST=0; shift ;;
		--no-release) DO_RELEASE=0; shift ;;
		-h|--help) sed -n '2,11p' "$0"; exit 0 ;;
		*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
done

PYTHON="${PYTHON:-python3}"
VERSION="$("$PYTHON" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
TAG="${TAG:-v${VERSION}}"
OUT="dist"
mkdir -p "$OUT"

sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | awk '{print $1}'
	else
		shasum -a 256 "$1" | awk '{print $1}'
	fi
}

# macOS 14.0 is the minimum imposed by the bundled ONNX Runtime; scikit-build-core
# derives the wheel tag from this variable.
if [[ "$(uname -s)" == "Darwin" ]]; then
	export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
fi

if [[ -n "$(git status --porcelain)" ]]; then
	echo "warning: the working tree has uncommitted changes; the release tag will point at HEAD" >&2
fi

echo "==> Building cpp-onnx-python ${VERSION} macOS wheel"
rm -f "$OUT"/*macosx*.whl
"$PYTHON" -m pip wheel . --no-build-isolation -w "$OUT"
MAC_WHEEL="$(ls "$OUT"/*macosx*.whl)"
echo "==> built $(basename "$MAC_WHEEL")"

if [[ "$DO_TEST" == "1" ]]; then
	echo "==> Testing the wheel in a clean virtual environment"
	VENV="$(mktemp -d)"
	trap 'rm -rf "$VENV"' EXIT
	"$PYTHON" -m venv "$VENV"
	"$VENV/bin/python" -m pip install -q --upgrade pip
	"$VENV/bin/python" -m pip install -q numpy pytest "$MAC_WHEEL"
	(cd /tmp && "$VENV/bin/python" -m pytest "$REPO_ROOT/tests" -q)
fi

WHEELS=("$OUT"/*.whl)
echo "==> wheels to publish:"
for wheel in "${WHEELS[@]}"; do
	echo "    $(basename "$wheel")  sha256:$(sha256_of "$wheel")"
done

if [[ "$DO_RELEASE" != "1" ]]; then
	echo "==> release skipped"
	exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
	cat <<EOF
==> 'gh' is not installed; upload the wheels manually:
    https://github.com/${REPO_SLUG}/releases/new?tag=${TAG}
EOF
	exit 0
fi

NOTES="Prebuilt wheels for **CPython 3.12+** (stable ABI, one per platform):"
for wheel in "${WHEELS[@]}"; do
	NOTES+=$'\n'"- \`$(basename "$wheel")\` — sha256 \`$(sha256_of "$wheel")\`"
done
NOTES+=$'\n\nInstall the wheel for your platform, e.g.:\n\n```console\npython -m pip install https://github.com/'"${REPO_SLUG}"'/releases/download/'"${TAG}"'/'"$(basename "$MAC_WHEEL")"$'\n```'

if gh release view "$TAG" >/dev/null 2>&1; then
	echo "==> uploading to existing release ${TAG}"
	gh release upload "$TAG" "${WHEELS[@]}" --clobber
else
	echo "==> creating release ${TAG}"
	gh release create "$TAG" "${WHEELS[@]}" --title "cpp-onnx-python ${TAG}" --notes "$NOTES"
fi

echo "==> done: https://github.com/${REPO_SLUG}/releases/tag/${TAG}"
