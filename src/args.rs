use clap::Parser;

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    long_about = "This program reads a folder with subfolders of JSONL files and 
outputs a folder of JSONL file with the threads and their posts."
)]
pub struct Cli {
    #[clap(short, long, help="Input to the root folder, internally must be in format path/*.jsonl",
    value_hint=clap::ValueHint::DirPath)]
    pub input: String,
    #[clap(short, long, help = "Output folder for the JSONL files, will write the jsonl as their own files",
    value_hint=clap::ValueHint::DirPath)]
    pub output: String,
    #[clap(
        short,
        long,
        help = "Tokenizer name: Accepts huggingface <org>/<name> or a path to tokenizer.json\nIf not provided, will split and count words"
    )]
    pub tokenizer: String,
}