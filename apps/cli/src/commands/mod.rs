use std::path::PathBuf;

use nixe_config::NixeConfig;

use crate::logging::{self, LogLevel};

pub mod list;
pub mod run;

fn load_config(
    path: Option<PathBuf>,
    log_level_override: Option<LogLevel>,
) -> Result<NixeConfig, String> {
    let config = match path {
        Some(path) => NixeConfig::load(path).map_err(|error| error.to_string()),
        None => NixeConfig::load_discovered()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "no configuration found; pass --config or create nixe.toml".to_owned()),
    }?;
    logging::set_level(log_level_override.unwrap_or_else(|| config.diagnostics.log_level.into()));
    Ok(config)
}
