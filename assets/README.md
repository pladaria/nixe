# System font replacements

`system.font` in `nixe.toml` selects the TTF served to applications by the shared-font service.
The files are distributed unchanged with their respective licenses.

- **DejaVu Sans 2.37** (default): the unchanged `ttf/DejaVuSans.ttf` from the
  [official release](https://dejavu-fonts.github.io/Download.html), package `dejavu-sans-ttf-2.37.zip`.
  License: `DejaVu-LICENSE.txt` (Bitstream Vera and Arev licenses, with public-domain DejaVu changes).
  TTF SHA-256: `7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954`.
  Archive SHA-256: `5c6e497a2f36552cb5ffb112c413a6af39c0f3c47653662b90b4fa6499822fd7`.
- **Open Sans Regular 3.003**: the complete `fonts/noto-set/ttf/OpenSans-Regular.ttf`
  from the [upstream Open Sans project](https://github.com/googlefonts/opensans/blob/bd7e37632246368c60fdcbd374dbf9bad11969b6/fonts/noto-set/ttf/OpenSans-Regular.ttf)
  at commit `bd7e37632246368c60fdcbd374dbf9bad11969b6`, rather than the Google Fonts distribution.
  License: `OpenSans-OFL.txt` (SIL Open Font License 1.1), from the same commit.
  SHA-256: `ce57ff2cbd11a2ce997256c3ec5d7c9e8a8199141bd7aaceda77f37fbecd71b3`.
- **Roboto Regular 3.015**: `hinted/static/Roboto-Regular.ttf` from the
  [official upstream release](https://github.com/googlefonts/roboto-3-classic/releases/tag/v3.015).
  License: `Roboto-OFL.txt` (SIL Open Font License 1.1).
  SHA-256: `58ef696cb53697e15c857c55aa528b06417e7396347c800edf166fedcbc85e35`.

DejaVu Sans maps U+25A1 (WHITE SQUARE); the bundled Open Sans and Roboto do not.
These replacements do not provide complete coverage of the console's system fonts.
