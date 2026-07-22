use std::path::PathBuf;

use swiitx_config::SwiitxConfig;

pub mod list;
pub mod run;

fn load_config(path: Option<PathBuf>) -> Result<SwiitxConfig, String> {
    match path {
        Some(path) => SwiitxConfig::load(path).map_err(|error| error.to_string()),
        None => SwiitxConfig::load_discovered()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                "no configuration found; pass --config or create swiitx.toml".to_owned()
            }),
    }
}
