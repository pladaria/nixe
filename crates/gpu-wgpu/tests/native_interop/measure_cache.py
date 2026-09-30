"""Compare fresh-process first submits with/without the native pipeline-cache file.

Pass the release native_interop test executable. Uses only temporary test caches;
the wgpu cache is identically seeded for both conditions. Driver/OS caches are
not reset, and whole-submit CPU times do not identify opaque driver cache hits.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile


def run(executable, directory):
    environment = os.environ.copy()
    assert not environment.get("NIXE_TEST_CAPTURE_DIR"), "run without capture instrumentation"
    assert "renderdoc" not in environment.get("LD_PRELOAD", "").lower()
    environment["NIXE_TEST_NATIVE_CACHE_DIR"] = str(directory)
    result = subprocess.run(
        [str(executable), "native_pipeline_cache_measurements", "--ignored", "--nocapture", "--exact"],
        env=environment, capture_output=True, text=True, timeout=60, check=True,
    )
    output = result.stdout + result.stderr
    assert "1 passed" in output, output
    duration = re.search(r"CACHE_MEASURE submission=1 submit_ns=(\d+)", output)
    assert duration, output
    return int(duration[1]) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    assert args.pairs >= 2
    executable = args.executable.resolve(strict=True)
    samples = {"native_absent": [], "native_present": []}
    with tempfile.TemporaryDirectory(prefix="nixe-cache-compare-") as temporary:
        root = Path(temporary)
        seed = root / "seed"
        seed.mkdir()
        run(executable, seed)
        wgpu = list(seed.glob("wgpu-*.bin"))
        native = list(seed.glob("native-vulkan-*.bin"))
        assert len(wgpu) == len(native) == 1, "expected both driver caches from seed run"
        for pair in range(args.pairs):
            # Balance order effects without letting either condition rewrite the
            # other's seed. Every invocation creates a new device/process.
            order = list(samples)
            if pair % 2:
                order.reverse()
            for mode in order:
                directory = root / (str(pair) + "-" + mode)
                directory.mkdir()
                for source in wgpu + (native if mode == "native_present" else []):
                    shutil.copyfile(source, directory / source.name)
                value = run(executable, directory)
                samples[mode].append(value)
                print(json.dumps({"pair": pair, "condition": mode, "submit_ms": value}), flush=True)
    print(json.dumps({
        mode: {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}
        for mode, values in samples.items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
