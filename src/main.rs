use std::fs;

use clap::Parser;
use rayon::prelude::*;
use std::path::Path;

pub mod args;
pub mod globals;
pub mod template;
pub mod conversations;
pub mod binpacking;
pub mod config;

fn main() -> std::io::Result<()> {

    let args = args::Cli::parse();
    let folder: String = args.input;
    let out_folder: String = args.output;
    // check if output folder exists
    if !Path::new(&out_folder).exists() {
        fs::create_dir(&out_folder)?;
    }
    let tokenizer: String = args.tokenizer;
    // let tokenizer: String = "aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct".to_string();

    globals::init_tokenizer(&tokenizer);
    // read config

    let config: config::TokenizerConfig = config::read_config(&tokenizer).unwrap();
    let template = template::ChatTemplate::new(
        config.chat_template,
        Some("<|begin_of_text|>".to_string()),
        Some("<|eot_id|>".to_string()),
    );

    let paths = fs::read_dir(folder)?;
    paths // filter only jsonl files
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "jsonl" {
                Some(path)
            } else {
                None
            }
        })
        .map(|path| {
            globals::TOTAL_JSONL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            path
        })
        .collect::<Vec<_>>()
        .par_iter()
        .map(|path| {
            let path = path.to_str().unwrap();
            conversations::single_jsonl_process(path, &out_folder, template.clone())
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    
Ok(())
}

#[cfg(test)]
mod tests {
    use tokenizers::Tokenizer;

    use super::*;

    #[test]
    fn test_template() {
        let tokenizer_conf_json = fs::read_to_string("tokenizer_config.json").unwrap();
        let config: config::TokenizerConfig = serde_json::from_str(&tokenizer_conf_json).unwrap();
        let template = template::ChatTemplate::new(
            config.chat_template,
            Some("<|begin_of_text|>".to_string()),
            Some("<|eot_id|>".to_string()),
        );
        let messages = vec![
            template::TextMessage {
                role: "user".to_string(),
                content: "What is the capital of Singapore?".to_string(),
            },
            template::TextMessage {
                role: "assistant".to_string(),
                content: "I don't know, what is it?".to_string(),
            },
        ];
        let result = template.apply(messages).unwrap();
        println!("{}", result);
    }
    #[test]
    fn test_tokenizer() {
        let tokenizer: String = "aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct".to_string();
        globals::init_tokenizer(&tokenizer);
        let content = "Hello world";
        let encoding = globals::tokenize(content);
        println!("{:?}", encoding.get_ids());
    }
    #[test]
    fn test_find_config() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let encoding = tokenizer.encode("Hey there!", false).unwrap();
        println!("{:?}", encoding.get_tokens());
    }
}