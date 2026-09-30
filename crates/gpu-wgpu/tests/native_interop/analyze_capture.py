"""Run with qrenderdoc --python; inspect opt-in native_interop captures only.

Replay timings are per-action GPU timestamps, not live application frame times.
References: https://renderdoc.org/docs/python_api/renderdoc/replay.html
            https://renderdoc.org/docs/python_api/renderdoc/counters.html
"""
import collections
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

import renderdoc as rd

TESS_STAGES = 0x2 | 0x4  # Vulkan tessellation control/evaluation stage flags.


def walk(actions):
    for action in actions:
        yield action
        yield from walk(action.children)


def analyze(path):
    capture = rd.OpenCaptureFile()
    controller = None
    try:
        status = capture.OpenFile(str(path), "", None)
        assert status == rd.ResultCode.Succeeded, str(status)
        status, controller = capture.OpenCapture(rd.ReplayOptions(), None)
        assert status == rd.ResultCode.Succeeded, str(status)
        structured = controller.GetStructuredFile()
        actions = list(walk(controller.GetRootActions()))
        events = {event.eventId: event for action in actions for event in action.events}
        calls = collections.Counter(structured.chunks[e.chunkIndex].name for e in events.values())
        # Resource creation chunks also include pre-capture initial state.
        # They are NOT evidence of cache misses in a warm captured submission.
        pipeline_creations = []
        for chunk in structured.chunks:
            if chunk.name == "vkCreateGraphicsPipelines":
                stages = chunk.FindChild("CreateInfo").FindChild("pStages")
                native = any(
                    stages.GetChild(i).FindChild("stage").AsInt() & TESS_STAGES
                    for i in range(stages.NumChildren())
                )
                pipeline_creations.append({
                    "kind": "native" if native else "ordinary",
                    "duration_us": chunk.metadata.durationMicro,
                })
        barriers = []
        for event_id, event in sorted(events.items()):
            chunk = structured.chunks[event.chunkIndex]
            if chunk.name == "vkCmdPipelineBarrier":
                barriers.append({
                    "event": event_id,
                    "src_stage": chunk.FindChild("srcStageMask").AsInt(),
                    "dst_stage": chunk.FindChild("destStageMask").AsInt(),
                })
        resources = {r.resourceId: r.name for r in controller.GetResources()}
        copies = [
            {"event": a.eventId, "source": resources.get(a.copySource, ""), "destination": resources.get(a.copyDestination, "")}
            for a in actions if a.flags & rd.ActionFlags.Copy
        ]
        draws = {}
        uniforms = set()
        for action in actions:
            if action.flags & rd.ActionFlags.Drawcall:
                controller.SetFrameEvent(action.eventId, True)
                native = controller.GetPipelineState().GetShader(rd.ShaderStage.Tess_Eval) != rd.ResourceId.Null()
                draws[action.eventId] = "native" if native else "ordinary"
                for binding in controller.GetPipelineState().GetConstantBlocks(rd.ShaderStage.Fragment):
                    uniforms.add(resources.get(binding.descriptor.resource, ""))
        counter = rd.GPUCounter.EventGPUDuration
        assert counter in controller.EnumerateCounters(), "GPU timestamps unavailable"
        totals = collections.defaultdict(list)
        for iteration in range(20):
            results = controller.FetchCounters([counter])
            sample = collections.defaultdict(float)
            observed = set()
            for result in results:
                if result.eventId in draws:
                    assert result.eventId not in observed, "duplicate draw timestamp"
                    observed.add(result.eventId)
                    assert math.isfinite(result.value.d) and result.value.d >= 0, "invalid GPU duration"
                    sample[draws[result.eventId]] += result.value.d * 1e6
            assert observed == set(draws), "missing draw timestamps"
            if iteration >= 3:
                for kind, duration in sample.items():
                    totals[kind].append(duration)
        return {
            "capture": path.name,
            "renderdoc": rd.GetVersionString(),
            "replay_samples": 17,
            "calls": dict(sorted(calls.items())),
            "pipeline_creations_including_initial_state": pipeline_creations,
            "draws": dict(collections.Counter(draws.values())),
            "barriers": barriers,
            "copies": copies,
            "fragment_uniform_buffers": sorted(uniforms),
            "draw_gpu_us": {
                kind: {"min": min(values), "median": statistics.median(values), "max": max(values)}
                for kind, values in totals.items()
            },
        }
    finally:
        if controller is not None:
            controller.Shutdown()
        capture.Shutdown()


try:
    directory = Path(os.environ["NIXE_TEST_CAPTURE_DIR"])
    paths = sorted(directory.glob("*_capture.rdc"))
    assert paths, "no captures found"
    reports = {}
    for path in paths:
        report = analyze(path)
        reports[path.name] = report
        os.write(1, (json.dumps(report, sort_keys=True) + "\n").encode())
    if "native_capture.rdc" in reports and "native32_capture.rdc" in reports:
        small = reports["native_capture.rdc"]
        large = reports["native32_capture.rdc"]
        assert small["draws"]["native"] == 2 and large["draws"]["native"] == 32
        for command in ("vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdBindDescriptorSets"):
            assert small["calls"][command] == large["calls"][command], "per-draw state/barrier growth: " + command
        assert small["calls"]["vkQueueSubmit"] == large["calls"]["vkQueueSubmit"] == 1
except BaseException:
    os.write(2, traceback.format_exc().encode())
    # qrenderdoc treats all SystemExit values as a successful script exit.
    # Replay owners were closed by finally; preserve failure for automation.
    os._exit(1)
# The script runs before qrenderdoc opens its UI. All replay owners are already
# shut down; exit instead of leaving a window/event loop behind in automation.
sys.exit(0)
