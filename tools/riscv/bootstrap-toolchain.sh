#!/bin/sh
# bootstrap-toolchain.sh - fetch the tools the RISC-V backend's build and
# tests use into tools-local/ (outside the repository; nothing is vendored).
#
#   Zig 0.17.0           cross compiler for riscv64-linux-musl / -gnu
#   qemu-user 10         riscv64 user-mode emulator for the regression tests
#
# Usage: tools/riscv/bootstrap-toolchain.sh [DEST_DIR]
# Then:  export PATH="$DEST_DIR/zig-x86_64-linux-0.17.0:$PATH"
#        zig build riscv-qemu-test -Dqemu="$DEST_DIR/qemu-user/usr/bin/qemu-riscv64"
set -eu

DEST="${1:-tools-local}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
ZIG_VERSION=0.17.0
ZIG_TAR="zig-x86_64-linux-$ZIG_VERSION.tar.xz"

if [ ! -x "$DEST/zig-x86_64-linux-$ZIG_VERSION/zig" ]; then
    echo "fetching Zig $ZIG_VERSION (resumable)"
    curl -fL -C - -o "$DEST/$ZIG_TAR" "https://ziglang.org/download/$ZIG_VERSION/$ZIG_TAR"
    tar -xJf "$DEST/$ZIG_TAR" -C "$DEST"
fi

# Debian's qemu-user is a static-pie binary that runs on any x86_64 glibc.
QEMU_DEB="qemu-user_10.0.13+ds-0+deb13u1_amd64.deb"
if [ ! -x "$DEST/qemu-user/usr/bin/qemu-riscv64" ]; then
    echo "fetching qemu-user 10.0.13 (riscv64 emulator)"
    curl -fL -o "$DEST/$QEMU_DEB" \
        "http://deb.debian.org/debian/pool/main/q/qemu/$QEMU_DEB"
    mkdir -p "$DEST/qemu-user"
    dpkg-deb -x "$DEST/$QEMU_DEB" "$DEST/qemu-user"
fi

echo "zig:   $DEST/zig-x86_64-linux-$ZIG_VERSION/zig"
echo "qemu:  $DEST/qemu-user/usr/bin/qemu-riscv64"
