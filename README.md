# collate-rs

Pet project on learning the data collation for huggingface [link](https://huggingface.co/blog/packing-with-FA2)

[In progress] I have added a python reference here for the implementation. 

## Pre-requisites

- Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

This works for non root users as well.
- Other packages (Linux tested)
```bash
sudo apt install libssl-dev pkg-config build-essential
```

Single block:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install libssl-dev pkg-config build-essential
```

### Installing to current directory

```bash
cargo install --git <this-repo> --branch main
```
## Usage

Preprocessing step:

```bash
cargo run --release -- --help

This program reads a folder with jsonl files and packs them into a msgpack for python.

Usage: collate [OPTIONS] --input <INPUT> --output <OUTPUT> --tokenizer <TOKENIZER> --format <FORMAT>

Options:
  -i, --input <INPUT>
          Input to the root folder, should contain jsonl files like so - path/*.jsonl or just a single file

  -o, --output <OUTPUT>
          Output folder for the JSONL files, will write the jsonl as their own files
              in the output folder. Eg. input/file.jsonl -> output/file.msgpack

  -t, --tokenizer <TOKENIZER>
          Accepts huggingface <org>/<name> format for the tokenizer

  -f, --format <FORMAT>
          Format of output file, [jsonl,arrow,msgpack]

          [default: arrow]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

```bash
cargo run --release -- -i data/ -o output/ -t mlx-community/Llama-3.2-1B-Instruct-4bit -f arrow
```

### Loading from python

The preferred method is to use arrow format, as it is the most performant. It can be read directly with datasets library.

```python
# currently the fields are hardcoded for training, but can be modified to suit the needs
# also assume the data has correct fields
from datasets import Dataset
dataset = Dataset.from_file("output/file.arrow")
```

## Issues and caveats
- Only tokenizers with chat_template, bos_token, eos_token are supported
- The format of the jsonl must contain a field called conversation, which is a list of dict with keys content and role

## Roadmap
[x] Integrate with python directly with Maturin - Completed but not very performant  
[ ] Add more tests  
[ ] Python reference code - For understanding  
[ ] Reduce memory overhead  
[ ] Implement mmap  