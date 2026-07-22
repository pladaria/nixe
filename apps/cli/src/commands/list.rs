use std::path::PathBuf;

use nixe_cli::library::Library;

use super::load_config;

pub struct Arguments {
    pub config_path: Option<PathBuf>,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    let config = load_config(arguments.config_path)?;
    let library = Library::scan(&config)?;

    for title in library.titles() {
        println!("{}\t{}", title.identifier, title.name);
    }
    Ok(())
}
