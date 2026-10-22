#!/usr/bin/env bash
# Build a private Linux Vulkan loader without installing anything system-wide.
# https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.4.345/BUILD.md
set -euo pipefail

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && $1 == --help ]]; then
        echo "Usage: $0"
        echo "Build pinned Vulkan Loader/Headers 1.4.345 in dev-tools/vulkan-loader/."
        echo "Includes the local ICD discovery synchronization patch."
        echo "Requires Linux, Git, CMake >= 3.22.1, Ninja, Python 3, a C compiler,"
        echo "pkg-config and X11/XCB/Xrandr development packages."
        exit 0
    fi
    echo "Usage: $0 [--help]" >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    echo "This recipe builds the Linux Vulkan loader only." >&2
    exit 1
fi

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
loader_root="$repository_root/dev-tools/vulkan-loader"
install_root="$loader_root/install"
# Includes the shared-list synchronization fix used by debug-utils calls.
# https://github.com/KhronosGroup/Vulkan-Loader/commit/5ee27b30c08505b868cb17a0683291d3a1d4543d
version=1.4.345
loader_revision=363f465abadab0a8dcfc5c85d2c691e9b0b788d6
headers_revision=74d8a6cb930c68ef617b202c3ff3c59d919e086b

checkout_source() {
    local name=$1 revision=$2
    local source="$loader_root/src/$name"
    if [[ ! -d $source ]]; then
        git init --quiet "$source"
        git -C "$source" fetch --depth=1 "https://github.com/KhronosGroup/$name.git" "$revision"
        git -C "$source" checkout --quiet --detach "$revision"
    fi
    if [[ $(git -C "$source" rev-parse HEAD) != "$revision" ||
          -n $(git -C "$source" status --porcelain) ]]; then
        echo "$source must be an unmodified checkout of $revision." >&2
        echo "Move the existing source directory aside, then rerun this script." >&2
        exit 1
    fi
}

checkout_source Vulkan-Headers "$headers_revision"
checkout_source Vulkan-Loader "$loader_revision"

# The startup reproducer is dev-tools/vulkan-startup-race.c. Keep the upstream
# checkout pristine between builds and record the exact distributed modification.
loader_patch="$repository_root/scripts/patches/vulkan-loader-serialize-icd-discovery.patch"
git -C "$loader_root/src/Vulkan-Loader" apply --check "$loader_patch"
git -C "$loader_root/src/Vulkan-Loader" apply "$loader_patch"
trap 'git -C "$loader_root/src/Vulkan-Loader" apply --reverse "$loader_patch"' EXIT

cmake -S "$loader_root/src/Vulkan-Headers" -B "$loader_root/build/headers" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_root" \
    -DVULKAN_HEADERS_ENABLE_TESTS=OFF \
    -DVULKAN_HEADERS_ENABLE_INSTALL=ON \
    -DVULKAN_HEADERS_ENABLE_MODULE=OFF
cmake --build "$loader_root/build/headers" --parallel
cmake --install "$loader_root/build/headers"

cmake -S "$loader_root/src/Vulkan-Loader" -B "$loader_root/build/loader" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_root" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_SYSCONFDIR=/etc \
    -DVulkanHeaders_DIR="$install_root/share/cmake/VulkanHeaders" \
    -DUPDATE_DEPS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_WERROR=OFF \
    -DBUILD_WSI_XLIB_SUPPORT=ON \
    -DBUILD_WSI_XCB_SUPPORT=ON \
    -DBUILD_WSI_XLIB_XRANDR_SUPPORT=ON \
    -DBUILD_WSI_WAYLAND_SUPPORT=ON \
    -DBUILD_WSI_DIRECTFB_SUPPORT=OFF
cmake --build "$loader_root/build/loader" --parallel
cmake --install "$loader_root/build/loader"

mkdir -p "$install_root/share/licenses/Vulkan-Loader" "$install_root/share/licenses/Vulkan-Headers"
cp "$loader_root/src/Vulkan-Loader/LICENSE.txt" "$install_root/share/licenses/Vulkan-Loader/"
cp "$loader_patch" "$install_root/share/licenses/Vulkan-Loader/"
cp "$loader_root/src/Vulkan-Headers/LICENSE.md" "$install_root/share/licenses/Vulkan-Headers/"
cp -R "$loader_root/src/Vulkan-Headers/LICENSES" "$install_root/share/licenses/Vulkan-Headers/"
cat > "$install_root/build-info.txt" <<EOF
Vulkan Loader/Headers: $version
Loader source: https://github.com/KhronosGroup/Vulkan-Loader
Loader commit: $loader_revision
Local patch: scripts/patches/vulkan-loader-serialize-icd-discovery.patch
Headers source: https://github.com/KhronosGroup/Vulkan-Headers
Headers commit: $headers_revision
Build recipe: scripts/build-vulkan-loader.sh
Configuration: Release, shared library, Xlib/XCB/Xrandr/Wayland WSI
EOF
uname -sm >> "$install_root/build-info.txt"
cmake --version >> "$install_root/build-info.txt"
sha256sum "$install_root/lib/libvulkan.so.$version" >> "$install_root/build-info.txt"
sha256sum "$loader_patch" >> "$install_root/build-info.txt"

echo "Vulkan loader installed locally: $install_root/lib/libvulkan.so.1"
echo "Cargo's Linux runner will use this copy for run/test/bench."
