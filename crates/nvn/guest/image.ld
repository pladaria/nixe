/* Position-independent code and immutable data share the RX mapping. The
 * following page holds the SDK version discovered by the resolver. */
ENTRY(nixe_nvn_error)
SECTIONS {
    . = 0;
    .text : { KEEP(*(.text .text.*)) }
    .rodata : { *(.rodata .rodata.*) }
    .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) *(.got .got.*) }
    .nvn_version ALIGN(4096) (NOLOAD) : { *(.nvn_version) }
    /DISCARD/ : { *(.eh_frame*) *(.comment) *(.note*) *(.gnu*) }
}
