# Nintendo Switch 1 Content Formats

This document describes the publicly documented relationships between the main
Nintendo Switch 1 content formats. It is a format and terminology reference,
not a description of any particular loader or implementation.

Public documentation for these formats is largely the result of community
reverse engineering. The references at the end of this document are therefore
part of the definition's provenance. Where the format permits several kinds of
content, this document describes the normal retail arrangement rather than
assuming that every file has the same layout.

## The short version

There are two common ways of distributing a title:

```text
Digital distribution
NSP
└── PFS0
    ├── Program NCA
    ├── Control NCA
    ├── Meta NCA
    ├── Patch NCA             (when an update is present)
    └── Add-on content NCAs   (when DLC is present)
```

```text
Game card
XCI
└── HFS0 card partitions
    ├── Normal
    ├── Logo
    ├── Update
    └── Secure
        └── NCAs
```

An NCA is the content archive that carries the actual title content. A program
NCA normally exposes two important sections:

```text
Program NCA
├── ExeFS section (PFS0)
│   ├── main       ── NSO executable
│   ├── main.npdm  ── process metadata
│   └── other executable files and libraries
└── RomFS section
    └── read-only program data and resources
```

The important distinction is that these names describe different layers. For
example, `NSP` is a distribution package, `NCA` is a content archive, `PFS0`
is a file container/filesystem, `RomFS` is a read-only filesystem, and `NSO`
is an executable format.

## Distribution formats

### NSP

`NSP` is the package form used for downloadable and installable content. Its
root is commonly a `PFS0` container. The entries are usually NCAs and may also
include related metadata such as tickets or certificates, depending on the
source and type of package.

An NSP does not itself define the executable or the game-data filesystem. It
groups the NCAs and associated files needed to represent a title, an update, or
add-on content.

Conceptually:

```text
NSP = PFS0 package containing one or more title-related files
```

### XCI

`XCI` is the image format for a Nintendo Switch game card. It has card-specific
headers and a card filesystem based on `HFS0`. The root card layout points to
partitions such as `normal`, `logo`, `update`, and `secure`.

The secure partition normally contains the NCAs required by the game card,
including its metadata NCA and program content. Therefore, an XCI is not just
an NSP with a different extension: it preserves the game-card layout and
card-specific structures.

The relationship is:

```text
XCI → HFS0 card layout → partition → NCA
```

`HFS0` is mentioned here because it is the filesystem used by the game-card
partition layout. It should not be confused with `PFS0`: both are related
Nintendo filesystem/container formats, but they have different headers and
roles.

## Containers and filesystems

### PFS0

`PFS0` is a simple table-based file container. Its header identifies the number
of entries and the size of a string table; each entry describes a file's offset,
size, and name. It is not limited to one kind of payload.

The same format can occur at different layers:

```text
NSP root       → PFS0 containing NCAs and package files
NCA section    → PFS0 containing files for that section
ExeFS section  → PFS0 containing executable-related files
```

Consequently, seeing the `PFS0` magic only identifies the immediate container.
The meaning of its entries depends on where that PFS0 was found.

### HFS0

`HFS0` is the hierarchical/card-oriented filesystem used by XCI. The root
HFS0 describes the card partitions, while partition HFS0 instances describe
files within those partitions. Its entries include file hashes and hashed
region information used by the card layout.

HFS0 is relevant to the XCI layer; it is not the usual root format of an NSP.

### NCA

`NCA` means Nintendo Content Archive. It is the signed and, in retail form,
encrypted content container used for title content. The NCA header identifies
properties such as distribution type, content type, title/program identity,
size, and the layout of its sections.

An NCA can contain several filesystem sections. The section header identifies
whether a section is a `RomFS` filesystem or a partition filesystem (`PFS0`).
Typical content types include:

| NCA content type | Typical role |
| --- | --- |
| Program | Executables and program data |
| Meta | CNMT metadata |
| Control | NACP application control data and icons |
| Data | System or title data |
| Manual | Manual content |
| Public data | Publicly accessible title data |

The exact combination depends on the title and content type. In particular,
not every NCA is a program NCA and not every NCA contains both ExeFS and
RomFS.

### ExeFS

`ExeFS` is the executable filesystem exposed by a program NCA. Its on-disk
section is a `PFS0` filesystem, but the name `ExeFS` describes the role of the
mounted content rather than a separate magic value.

A program ExeFS commonly contains:

- `main`, the program executable;
- `main.npdm`, process and permission metadata;
- additional modules or libraries, often represented by NSO files; and
- other executable-related metadata.

The `ExeFS` section is therefore a PFS0 container at a particular location in
the NCA hierarchy.

### RomFS

`RomFS` is the read-only hierarchical filesystem used for title resources and
other immutable data. It has its own filesystem header, directory metadata,
file metadata, hash tables, and file-data region.

A program's RomFS commonly contains assets, configuration, localization,
shaders, and game data. RomFS is not an archive of NCAs and is not the same
thing as the outer NSP/XCI distribution format.

Updates can use NCA metadata and patch-specific structures to describe how
updated content relates to base content. An update is therefore a title-level
relationship between content archives, not simply a replacement of the outer
NSP or XCI container.

## Executable formats

### NSO

`NSO` is the main executable format used by official Switch software. It starts
with the `NSO0` signature and describes the `.text`, `.rodata`, and `.data`
segments, including their file and memory sizes. Segments may be compressed
with LZ4.

In the normal title layout, an NSO is a file inside ExeFS. `NSO` is therefore
below the NCA and ExeFS layers:

```text
NCA → ExeFS/PFS0 → NSO
```

### NRO

`NRO` is the executable format for binaries loaded outside ExeFS, most notably
homebrew applications. It starts with the `NRO0` signature and describes the
text, read-only, and data segments, as well as relocatable-object metadata and
optional embedded data.

An NRO is not the normal official-title equivalent of an NSO. The practical
distinction is:

| Format | Normal role | Normal location |
| --- | --- | --- |
| NSO | Official title executable | Inside a program's ExeFS |
| NRO | Homebrew or non-ExeFS executable | Loaded as an external binary |

Both formats describe executable images, but they belong to different loading
contexts and should not be treated as interchangeable container formats.

## Metadata and title relationships

### CNMT

`CNMT` is the packaged-content metadata format. Its official name is
`PackagedContentMeta`. A CNMT describes a title's content metadata, including
the content entries and relationships to other content metadata. It is
normally found in a meta NCA as a `.cnmt` file.

CNMT is what connects a group of NCAs to a title-level concept such as an
application, patch, add-on content, or system update. It is metadata about the
content set; it is not a replacement for the NCA payloads themselves.

### NACP

`NACP` is application-control metadata, normally found as `control.nacp` in a
control NCA. It contains user-facing and application-control information such
as title names, publisher information, supported languages, and related
application properties. Icons are commonly stored alongside it in the control
content.

CNMT and NACP have different responsibilities:

```text
CNMT → describes packaged content and title relationships
NACP → describes application-control and presentation metadata
```

## Complete relationship examples

### Downloaded application

```text
NSP
└── PFS0
    ├── Program NCA
    │   ├── ExeFS/PFS0
    │   │   ├── main (NSO)
    │   │   └── main.npdm
    │   └── RomFS
    ├── Control NCA
    │   └── control.nacp and icons
    └── Meta NCA
        └── .cnmt
```

### Game-card application

```text
XCI
└── HFS0
    └── Secure partition
        ├── Program NCA
        ├── Control NCA
        └── Meta NCA
```

The same kinds of NCAs can be reached through different distribution layers;
the outer container determines how they are packaged, while the NCA and its
sections determine how the title content is represented.

### Homebrew application

```text
NRO
└── executable image loaded outside the official program-ExeFS layout
```

An NRO may use additional files or embedded assets according to the homebrew
application's conventions. That does not make it an NSP, NCA, or RomFS.

## Terminology summary

| Term | Layer | Main purpose |
| --- | --- | --- |
| NSP | Distribution | Package downloadable/title content files |
| XCI | Distribution | Represent a game-card image and its partitions |
| HFS0 | Container/filesystem | Describe XCI card partitions and their files |
| NCA | Content archive | Carry signed/encrypted title content |
| PFS0 | Container/filesystem | Store a table of named files at several layers |
| ExeFS | NCA section role | Expose executable-related files, using PFS0 |
| RomFS | NCA section/filesystem | Expose read-only hierarchical data |
| CNMT | Metadata | Describe packaged-content and title relationships |
| NACP | Metadata | Describe application-control information |
| NSO | Executable | Official Switch executable image |
| NRO | Executable | Non-ExeFS/homebrew executable image |

## References

The following public technical references were consulted. Switchbrew is a
community-maintained reverse-engineering reference; Nintendo has not published
a complete public specification for all of these retail formats.

- [Switchbrew: NCA](https://switchbrew.org/wiki/NCA_Format)
- [Switchbrew: XCI](https://switchbrew.org/wiki/XCI)
- [Switchbrew: CNMT](https://switchbrew.org/wiki/CNMT)
- [Switchbrew: NCA Content FS](https://switchbrew.org/wiki/NCA_Content_FS)
- [Switchbrew: NSO](https://switchbrew.org/wiki/NSO)
- [Switchbrew: NRO](https://switchbrew.org/wiki/NRO)
- [Switchbrew: Homebrew ABI](https://switchbrew.org/wiki/Homebrew_ABI)
- [libnx NRO header reference](https://switchbrew.github.io/libnx/nro_8h.html)
