use clap::Parser;

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    long_about = "This program reads a folder with jsonl files and packs them into a msgpack for python."
)]
pub struct Cli {
    #[clap(short, long, help="Input to the root folder, should contain jsonl files like so - path/*.jsonl",
    value_hint=clap::ValueHint::DirPath)]
    pub input: String,
    #[clap(short, long, help = "Output folder for the JSONL files, will write the jsonl as their own files
    in the output folder. Eg. input/file.jsonl -> output/file.msgpack",
    value_hint=clap::ValueHint::DirPath)]
    pub output: String,
    #[clap(
        short,
        long,
        help = "Accepts huggingface <org>/<name> format for the tokenizer"
    )]
    pub tokenizer: String,
    #[clap(
        short,
        long,
        help = "Format of output file, [jsonl,arrow,msgpack]",
        default_value = "arrow"
    )]
    pub format: String,
}
