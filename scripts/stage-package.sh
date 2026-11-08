#!/bin/bash
# Lay out the installed file tree that every distribution package installs.
#
# One staging step shared by the Debian, Arch and RPM builds, so the three
# cannot drift into installing different things. It writes a plain filesystem
# tree under a destination root; each packaging format then only has to
# describe metadata and dependencies.
#
# Usage: stage-package.sh <destroot> [prefix]
set -euo pipefail

DESTROOT="${1:?usage: stage-package.sh <destroot> [prefix]}"
PREFIX="${2:-/usr}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="intel-npu-tools"
LIBDIR="$PREFIX/lib/$NAME"

# This script begins by deleting its destination, and the destination comes
# from whoever called it — %{buildroot}, $pkgdir, a mktemp directory. That is
# fine until one of those is empty or mistyped, at which point an rm -rf is
# pointed at something real. None of these refusals can trigger for a genuine
# staging directory, and each of them can for a mistake.
refuse() { echo "stage-package.sh: refusing to stage into '$DESTROOT': $1" >&2; exit 1; }
case "$DESTROOT" in
  /*) ;;
  *) refuse "the destination must be an absolute path" ;;
esac
# A build root is always several directories deep. A single component is /usr,
# /etc, /home or the root itself, and never a place to stage a package.
case "${DESTROOT#/}" in
  */*) ;;
  *) refuse "it is a top-level directory" ;;
esac
if [ -e "$DESTROOT/etc/fstab" ] || [ -e "$DESTROOT/proc/self" ]; then
  refuse "it looks like a running system rather than a build directory"
fi
if [ -d "$DESTROOT" ] && [ "$(cd -- "$DESTROOT" && pwd -P)" = "$PROJECT_DIR" ]; then
  refuse "it is the project directory"
fi

COMMANDS=(intel-npu-info intel-npu-mcp intel-npu-ocr intel-npu-speech intel-npu-search intel-npu-panel intel-npu-transcribe)
APPLICATIONS=(intel-npu-speech intel-npu-ocr intel-npu-panel)

rm -rf "$DESTROOT"
install -d "$DESTROOT$LIBDIR" "$DESTROOT$PREFIX/bin" \
           "$DESTROOT$PREFIX/share/applications" \
           "$DESTROOT$PREFIX/share/doc/$NAME"

# The importable source, its packaging metadata, and the model downloader the
# setup command runs. This is what the per-user environment is built from, so
# it has to be a working source tree rather than only the built modules.
cp -a "$PROJECT_DIR/src" "$DESTROOT$LIBDIR/src"
cp -a "$PROJECT_DIR/pyproject.toml" "$DESTROOT$LIBDIR/"
cp -a "$PROJECT_DIR/README.md" "$PROJECT_DIR/LICENSE" "$DESTROOT$LIBDIR/"
install -d "$DESTROOT$LIBDIR/scripts"
install -m755 "$PROJECT_DIR/scripts/download-models.py" "$DESTROOT$LIBDIR/scripts/"
find "$DESTROOT$LIBDIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$DESTROOT$LIBDIR" -name '*.egg-info' -type d -prune -exec rm -rf {} +

# One wrapper per command, plus the setup entry point.
for command in "${COMMANDS[@]}"; do
  sed -e "s|@COMMAND@|$command|g" "$PROJECT_DIR/packaging/wrapper.in" \
    > "$DESTROOT$PREFIX/bin/$command"
  chmod 755 "$DESTROOT$PREFIX/bin/$command"
done
install -m755 "$PROJECT_DIR/packaging/$NAME-setup" "$DESTROOT$PREFIX/bin/$NAME-setup"

# Desktop entries point at the packaged wrappers rather than a home directory.
for application in "${APPLICATIONS[@]}"; do
  sed -e "s|@BINDIR@|$PREFIX/bin|g" "$PROJECT_DIR/packaging/$application.desktop.in" \
    > "$DESTROOT$PREFIX/share/applications/$application.desktop"
  chmod 644 "$DESTROOT$PREFIX/share/applications/$application.desktop"
done

"$PROJECT_DIR/scripts/render-icons.sh" "$DESTROOT$PREFIX/share/icons/hicolor" >/dev/null

install -m644 "$PROJECT_DIR/README.md" "$PROJECT_DIR/CHANGELOG.md" "$DESTROOT$PREFIX/share/doc/$NAME/"
install -d "$DESTROOT$PREFIX/share/doc/$NAME/docs"
install -m644 "$PROJECT_DIR"/docs/*.md "$DESTROOT$PREFIX/share/doc/$NAME/docs/"

echo "Staged $NAME into $DESTROOT$PREFIX"
