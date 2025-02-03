use std::fs;

use clap::Parser;
use std::path::Path;

pub mod args;
pub mod binpacking;
pub mod config;
#[macro_use]
pub mod utils;
pub mod conversations;
pub mod globals;
pub mod template;

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
    let paths : Vec<String> = {
        let md = fs::metadata(&folder)?;
        if md.is_file() {
            vec![folder]
        } else {
            fs::read_dir(&folder)?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.extension()?.to_str()? == "jsonl" {
                        Some(path.into_os_string().into_string().unwrap())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        }
    };
    globals::TOTAL_JSONL.fetch_add(paths.len().try_into().unwrap(), std::sync::atomic::Ordering::SeqCst);
    paths.into_iter() // filter only jsonl files
        .for_each(|path| {
            let _ = conversations::single_jsonl_process(
                &path,
                &out_folder,
                template.clone(),
                args.format.clone(),
            );
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use tokenizers::Tokenizer;

    use super::*;

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
