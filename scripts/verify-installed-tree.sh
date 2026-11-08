#!/bin/bash
# Assert that an installed native package really put its files on the system.
#
# Run after installing the Debian, RPM or Arch package. Every check is reported
# rather than the script dying on the first one, because "the install is
# broken" is a much less useful answer than which of thirty files is absent.
#
# Deliberately does not run any of the installed commands. They bootstrap a
# per-user virtual environment and download models, which is exactly what
# installing the package is designed not to do.
#
# Usage: verify-installed-tree.sh [prefix]
set -euo pipefail

PREFIX="${1:-/usr}"
NAME="intel-npu-tools"
LIBDIR="$PREFIX/lib/$NAME"
failed=0

check() {
  local description="$1"; shift
  if "$@"; then
    echo "  ok       $description"
  else
    echo "  MISSING  $description"
    failed=1
  fi
}

echo "Commands:"
for command in intel-npu-info intel-npu-mcp intel-npu-ocr \
               intel-npu-speech intel-npu-search intel-npu-panel \
               intel-npu-transcribe "$NAME-setup"; do
  check "$PREFIX/bin/$command" test -x "$PREFIX/bin/$command"
done

echo "Library tree:"
# The importable source has to be a working tree, not only the built modules,
# because the per-user environment is installed from it.
check "$LIBDIR/src/intel_npu_tools/config.py" test -f "$LIBDIR/src/intel_npu_tools/config.py"
check "$LIBDIR/pyproject.toml"                test -f "$LIBDIR/pyproject.toml"
check "$LIBDIR/scripts/download-models.py"    test -x "$LIBDIR/scripts/download-models.py"
check "$LIBDIR/LICENSE"                       test -f "$LIBDIR/LICENSE"

echo "Documentation:"
check "$PREFIX/share/doc/$NAME/README.md"    test -f "$PREFIX/share/doc/$NAME/README.md"
check "$PREFIX/share/doc/$NAME/CHANGELOG.md" test -f "$PREFIX/share/doc/$NAME/CHANGELOG.md"

echo "Desktop integration:"
# The scalable icon is the one that is always installed; the rendered sizes
# depend on a rasteriser being present when the package was built, and a
# package built without one would quietly ship only the SVG.
check "$PREFIX/share/icons/hicolor/scalable/apps/$NAME.svg" \
  test -f "$PREFIX/share/icons/hicolor/scalable/apps/$NAME.svg"
check "$PREFIX/share/icons/hicolor/48x48/apps/$NAME.png" \
  test -f "$PREFIX/share/icons/hicolor/48x48/apps/$NAME.png"
for application in intel-npu-speech intel-npu-ocr intel-npu-panel; do
  entry="$PREFIX/share/applications/$application.desktop"
  check "$entry" test -f "$entry"
  if [ -f "$entry" ] && command -v desktop-file-validate >/dev/null; then
    check "$entry is valid" desktop-file-validate "$entry"
  fi
done

# The checks above name the files that matter, which is what catches a package
# that installs cleanly but forgot something. They cannot catch a file the
# package promised and did not deliver, because they are a hand-written list
# and the package's own manifest is the authority on that. So ask it.
echo "Every file the package says it installed:"
# Not tolerated if it fails. An unreadable manifest would otherwise leave
# nothing to compare against, and a check with nothing to compare against
# passes — which is the failure this whole section exists to prevent.
if ! manifest=$("$(dirname -- "${BASH_SOURCE[0]}")/package-files.sh" "$NAME" 2>&1); then
  echo "  MISSING  cannot read the package manifest: $manifest"
  failed=1
else
  absent=0
  total=0
  while read -r path; do
    [ -n "$path" ] || continue
    total=$((total + 1))
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then
      echo "  MISSING  $path"
      absent=$((absent + 1))
      failed=1
    fi
  done <<< "$manifest"
  if [ "$absent" -eq 0 ]; then
    echo "  ok       all $total of them are on disk"
  else
    echo "  $absent of $total are not on disk"
  fi
fi

echo "Nothing that should never be packaged:"
# Models are enormous and separately licensed; build residue is just sloppy.
# find's own failure is a failure here: discarding its errors would turn an
# unreadable directory into an empty result, which reads as nothing found.
no_such_file() {
  local hit
  hit=$(find "$1" -name "$2" -print -quit) || return 1
  [ -z "$hit" ]
}
for pattern in '*.bin' '__pycache__' '*.egg-info'; do
  check "no $pattern under $LIBDIR" no_such_file "$LIBDIR" "$pattern"
done

if [ "$failed" -ne 0 ]; then
  echo >&2
  echo "The installed tree is not what the package promised." >&2
  exit 1
fi

echo "The installed tree is complete."
