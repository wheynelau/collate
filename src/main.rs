use std::fs;

use clap::Parser;
use conversations::TokenizedInput;
use rayon::prelude::*;
use std::path::Path;

pub mod args;
pub mod globals;
pub mod template;
pub mod conversations;
pub mod binpacking;
pub mod config;

fn single_jsonl_process(jsonl_path: &str, out_folder: &str,template: template::ChatTemplate) -> std::io::Result<()> {
    let msg_pack_path = {
        let path = Path::new(jsonl_path);
        let file_stem = path.file_stem() // get the filename without extension
            .expect("Invalid file path")
            .to_str()
            .expect("Invalid file path");
        let mut out_path = Path::new(out_folder).join(file_stem);
        out_path.set_extension("msgpack");
        out_path
    };
    // read jsonl for testing
    let data = conversations::read_jsonl(jsonl_path, template);
    dbg!(&data.len());
    // parallelize the tokenization
    let mut inputs: Vec<TokenizedInput> = data
        .par_iter()
        .map(|conv| {
            let input_ids: Vec<u32> = globals::tokenize(conv).get_ids().to_owned();
            let mut labels: Vec<i32> = input_ids.iter().map(|x| *x as i32).collect();
            // replace labels.0 with -100
            labels[0] = -100;
            let position_ids = (0..input_ids.len() as u32).collect();
            let length = input_ids.len() as u32;
            conversations::TokenizedInput {
                input_ids,
                labels,
                position_ids,
                length,
            }
        })
        .collect();
    
    inputs.sort_by(|a, b| b.length.cmp(&a.length));
    println!("Last length: {}", inputs.last().unwrap().length);

    // bin packing
    let max_length = 8192;
    let bins = binpacking::create_bins(inputs, max_length);

    // serialize the bins
    let encoded = rmp_serde::to_vec(&bins).unwrap();
    fs::write(msg_pack_path, encoded)?;

    Ok(())

    }
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
        .collect::<Vec<_>>()
        .par_iter()
        .map(|path| {
            let path = path.to_str().unwrap();
            single_jsonl_process(path, &out_folder, template.clone())
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