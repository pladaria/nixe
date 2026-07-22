use std::path::PathBuf;

use nixe_config::NixeConfig;

pub mod list;
pub mod run;

fn load_config(path: Option<PathBuf>) -> Result<NixeConfig, String> {
    match path {
        Some(path) => NixeConfig::load(path).map_err(|error| error.to_string()),
        None => NixeConfig::load_discovered()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "no configuration found; pass --config or create nixe.toml".to_owned()),
    }
}
