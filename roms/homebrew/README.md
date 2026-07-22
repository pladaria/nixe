# Homebrew test assets

The `.nro` files are redistributable builds from
<https://github.com/switchbrew/switch-examples> and exercise contemporary
libnx startup paths.

`acceptance/minimal-a64.nro.fixture` is a declarative, reviewable NRO byte
image owned by this project. It enters through the real Homebrew ABI, executes
`GetCurrentProcessorNumber`, and returns through the loader link register. The
test materializes unspecified bytes as zero; this keeps the committed fixture
small without hiding generated binary contents.
