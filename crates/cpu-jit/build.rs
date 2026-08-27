fn main() {
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rerun-if-changed=src/fastmem_fault.c");
        cc::Build::new()
            .file("src/fastmem_fault.c")
            .flag_if_supported("-std=c11")
            .warnings(true)
            .compile("nixe_fastmem_fault");
    }
}
