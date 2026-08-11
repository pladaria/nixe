# nv_push_dump container

This image builds Mesa 26.0.6's Nouveau `nv_push_dump` tool with support for
the Maxwell classes used by Switch 1. The wrapper selects Mesa's `MAXWELL`
command-line profile by default.

Use the repository wrapper from any working directory:

```sh
scripts/nv-push-dump.sh ./dump/maxwell-b197-example.bin
```

The wrapper builds `nixe/nv-push-dump:mesa-26.0.6` on first use, mounts only
the input file's directory as read-only, and runs the container with the
caller's numeric user and group IDs. Output is written to stdout and can be
redirected normally:

```sh
scripts/nv-push-dump.sh ./dump/maxwell-b197-example.bin \
    > ./dump/maxwell-b197-example.txt
```

Mesa 26.0.6 does not include the Nouveau source directory when only
`tools=nouveau` is enabled. The Docker context carries a minimal patch that
includes it without enabling a Gallium or Vulkan driver. See Mesa's versioned
[`src/meson.build`](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.0.6/src/meson.build)
and [`src/nouveau/headers/meson.build`](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.0.6/src/nouveau/headers/meson.build).

Pass a second argument only to inspect a different NVIDIA architecture:

```sh
scripts/nv-push-dump.sh ./dump/example.bin KEPLER
```

Mesa documents the tool and its `-Dtools=nouveau` build option here:
<https://docs.mesa3d.org/drivers/nvk/external_hardware_docs.html>.
