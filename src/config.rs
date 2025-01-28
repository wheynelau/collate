/// Handles the downloading of configuration files
/// Reference from hf tokenizers for downloading:
/// https://github.com/huggingface/tokenizers/blob/c45aebd1029acfbe9e5dfe64e8b8441d9fae727a/tokenizers/src/utils/from_pretrained.rs#L26

use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
pub struct TokenizerConfig{
    pub bos_token: String,
    pub eos_token: String,
    pub chat_template: String,
}

/// Defines the additional parameters available for the `from_pretrained` function
#[derive(Debug, Clone)]
pub struct FromPretrainedParameters {
    pub revision: String,
    pub user_agent: HashMap<String, String>,
    pub token: Option<String>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            user_agent: HashMap::new(),
            token: None,
        }
    }
}

/// Downloads and cache the identified tokenizer if it exists on
/// the Hugging Face Hub, and returns a local path to the file
fn from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let identifier: String = identifier.as_ref().to_string();

    let valid_chars = ['-', '_', '.', '/'];
    let is_valid_char = |x: char| x.is_alphanumeric() || valid_chars.contains(&x);

    let valid = identifier.chars().all(is_valid_char);
    let valid_chars_stringified = valid_chars
        .iter()
        .fold(vec![], |mut buf, x| {
            buf.push(format!("'{}'", x));
            buf
        })
        .join(", "); // "'/', '-', '_', '.'"
    if !valid {
        return Err(format!(
            "Model \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            identifier
        )
        .into());
    }
    let params = params.unwrap_or_default();

    let revision = &params.revision;
    let valid_revision = revision.chars().all(is_valid_char);
    if !valid_revision {
        return Err(format!(
            "Revision \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            revision
        )
        .into());
    }

    let mut builder = ApiBuilder::new();
    if let Some(token) = params.token {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let repo = Repo::with_revision(identifier, RepoType::Model, params.revision);
    let api = api.repo(repo);
    Ok(api.get("tokenizer_config.json")?)
}

pub fn read_config(tokenizer: &str) -> Result<TokenizerConfig, Box<dyn std::error::Error>> {
    let path = from_pretrained(tokenizer, None)?;
    let config = std::fs::read_to_string(path)?;
    let config: TokenizerConfig = serde_json::from_str(&config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_config() {
        let tokenizer = "mlx-community/Llama-3.2-1B-Instruct-4bit";
        let config = read_config(tokenizer).unwrap();
        assert_eq!(config.bos_token, "<|begin_of_text|>");
        assert_eq!(config.eos_token, "<|eot_id|>");
    }
}

