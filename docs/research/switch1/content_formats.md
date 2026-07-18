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

Compressed variants of these distribution formats are also commonly used:

```text
NSZ → NSP/PFS0 layout with NCA payloads represented as NCZ
XCZ → XCI/HFS0 layout with NCA payloads represented as NCZ
```

`NCZ` is a compressed representation of an NCA. It is usually encountered as
an entry inside an NSZ or XCZ rather than as a standalone file selected by a
user. Decompressing an NCZ reconstructs the NCA byte stream expected by an NCA
reader; it does not introduce a new title-content model.

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

### NSZ and XCZ

`NSZ` and `XCZ` are compressed distribution formats. NSZ retains the package
role and PFS0-based organization of NSP, while XCZ retains the game-card role
and HFS0-based organization of XCI. Their NCA payloads are stored in NCZ form
to reduce their size.

Conceptually:

```text
NSZ = compressed NSP representation containing NCZ entries
XCZ = compressed XCI representation containing NCZ entries
```

Compression changes how the content bytes are stored, not the title-level
meaning of program, control, meta, patch, or add-on content. A loader can
therefore expose each NCZ as a reconstructed NCA byte stream and reuse the
same NCA, CNMT, NACP, ExeFS, and RomFS readers used for NSP and XCI.

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

| NCA content type | Typical role                            |
| ---------------- | --------------------------------------- |
| Program          | Executables and program data            |
| Meta             | CNMT metadata                           |
| Control          | NACP application control data and icons |
| Data             | System or title data                    |
| Manual           | Manual content                          |
| Public data      | Publicly accessible title data          |

The exact combination depends on the title and content type. In particular,
not every NCA is a program NCA and not every NCA contains both ExeFS and
RomFS.

### NCZ

`NCZ` is the compressed form used for an NCA payload in NSZ and XCZ. It keeps
the information needed to reconstruct the original NCA representation while
compressing the section data, normally with Zstandard. NCZ-specific section
metadata describes how the stored data maps back to NCA sections and their
encryption state.

NCZ should be treated as a storage transformation around NCA rather than as a
peer of Program, Control, or Meta content. Once reconstructed, its output is
consumed as an ordinary NCA:

```text
NCZ → decompression/reconstruction → NCA byte stream → NCA reader
```

Although a standalone `.ncz` file is possible, users normally see `.nsz` or
`.xcz` files because those outer formats package all files belonging to the
distribution. NCZ files are generally internal entries within them.

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

### BKTR and patch RomFS

`BKTR` is the magic used by the bucket-tree structures that describe a patch
RomFS. An update does not need to store another complete copy of the base
RomFS. Instead, BKTR metadata describes how to construct one effective virtual
image from ranges belonging to the base content and ranges stored in the
update.

Conceptually:

```text
base RomFS ranges ─────┐
                       ├─ BKTR relocation view ─ effective patched RomFS
update RomFS ranges ───┘
```

A patch RomFS uses two related BKTR tables:

- the relocation table maps each range of the effective virtual image to an
  offset in either the base image or the physical patch image; and
- the subsection table divides the physical patch image into regions and
  records the AES-CTR-Ex counter-generation value required to decrypt each
  region.

The relocation entries are ordered by their offsets in the patched image, so a
reader can locate the applicable mapping and forward a read to the appropriate
source. A read that crosses mapping boundaries may alternate between base and
update storage. Patch-data reads may also need to be split at subsection
boundaries because different subsections can use different counter-generation
values. The BKTR table data itself uses the section's normal encryption rather
than the per-subsection counters.

This makes BKTR suitable for a lazy storage view: the loader retains the small
validated mapping tables and reads the required bytes from the base or update
on demand. It does not need to allocate, extract, or write a complete patched
RomFS. Once the virtual image and its IVFC data level have been composed, the
result can be consumed by the same RomFS reader used for an unpatched title.

BKTR is therefore patch mapping metadata inside an update NCA, not another
distribution format like NSP or XCI and not a general-purpose replacement for
RomFS.

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

| Format | Normal role                      | Normal location              |
| ------ | -------------------------------- | ---------------------------- |
| NSO    | Official title executable        | Inside a program's ExeFS     |
| NRO    | Homebrew or non-ExeFS executable | Loaded as an external binary |

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

### Compressed application package

```text
NSZ/PFS0 or XCZ/HFS0
└── NCZ
    └── reconstructed NCA
        ├── ExeFS/PFS0
        ├── RomFS
        ├── control.nacp and icons
        └── .cnmt
```

The files present under an individual NCA depend on its content type; the
diagram combines the possible relationships to show that the layers below NCA
remain unchanged.

### Homebrew application

```text
NRO
└── executable image loaded outside the official program-ExeFS layout
```

An NRO may use additional files or embedded assets according to the homebrew
application's conventions. That does not make it an NSP, NCA, or RomFS.

## Terminology summary

| Term  | Layer                             | Main purpose                                           |
| ----- | --------------------------------- | ------------------------------------------------------ |
| NSP   | Distribution                      | Package downloadable/title content files               |
| XCI   | Distribution                      | Represent a game-card image and its partitions         |
| NSZ   | Compressed distribution           | Represent an NSP with NCA payloads stored as NCZ       |
| XCZ   | Compressed distribution           | Represent an XCI with NCA payloads stored as NCZ       |
| HFS0  | Container/filesystem              | Describe XCI card partitions and their files           |
| NCA   | Content archive                   | Carry signed/encrypted title content                   |
| NCZ   | Compressed content representation | Store a reconstructable, compressed NCA payload        |
| PFS0  | Container/filesystem              | Store a table of named files at several layers         |
| ExeFS | NCA section role                  | Expose executable-related files, using PFS0            |
| RomFS | NCA section/filesystem            | Expose read-only hierarchical data                     |
| BKTR  | Patch mapping metadata            | Compose an effective RomFS from base and update ranges |
| CNMT  | Metadata                          | Describe packaged-content and title relationships      |
| NACP  | Metadata                          | Describe application-control information               |
| NSO   | Executable                        | Official Switch executable image                       |
| NRO   | Executable                        | Non-ExeFS/homebrew executable image                    |

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
- [NSZ format specification and reference implementation](https://github.com/nicoboss/nsz)
