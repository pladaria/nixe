#!/usr/bin/env python3
"""Rebuild checked-in native helper modules with the pinned Vulkan SDK tools."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sdk-bin', type=Path, required=True, help='SDK 1.4.341.1 x86_64/bin directory')
parser.add_argument('--check', action='store_true', help='validate reproducibility without modifying artifacts')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / 'crates/gpu-vulkan/shaders'
lock = json.loads((root / 'toolchain.json').read_text())
for name, expected in lock['tools'].items():
    tool = args.sdk_bin / name
    if not tool.is_file() or hashlib.sha256(tool.read_bytes()).hexdigest() != expected:
        parser.error(f'{name} must match the pinned Linux x86_64 SDK {lock["sdk"]} binary')
with tempfile.TemporaryDirectory(prefix='nixe-helper-shaders-') as temporary:
    for source in sorted((root / 'src').iterdir()):
        if source.suffix not in ('.vert', '.frag', '.comp'):
            continue
        output = Path(temporary) / (source.name + '.spv')
        subprocess.run([str(args.sdk_bin / 'glslc'), '--target-env=' + lock['target'], '--target-spv=' + lock['spirv'], '-O', '-Werror', str(source), '-o', str(output)], check=True)
        subprocess.run([str(args.sdk_bin / 'spirv-val'), '--target-env', lock['target'], str(output)], check=True)
        destination = root / 'spv' / output.name
        data = output.read_bytes()
        if args.check:
            if not destination.is_file() or destination.read_bytes() != data:
                parser.error(f'stale helper module: {destination.relative_to(root)}')
        else:
            destination.write_bytes(data)
        print(f'{source.name}: validated, sha256={hashlib.sha256(data).hexdigest()}')
