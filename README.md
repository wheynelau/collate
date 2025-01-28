# collate-rs

Pet project on learning the data collation for huggingface [link](https://huggingface.co/blog/packing-with-FA2)

I have added a python reference [here](./python_reference.py) for the implementation.

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
cargo install <this-repo> .
```
## Usage

Preprocessing step:

```bash
cargo run --release -- --help

This program reads a folder with jsonl files and packs them into a msgpack for python.

Usage: collate --input <INPUT> --output <OUTPUT> --tokenizer <TOKENIZER>

Options:
  -i, --input <INPUT>
          Input to the root folder, should contain jsonl files like so - path/*.jsonl

  -o, --output <OUTPUT>
          Output folder for the JSONL files, will write the jsonl as their own files
              in the output folder. Eg. input/file.jsonl -> output/file.msgpack

  -t, --tokenizer <TOKENIZER>
          Accepts huggingface <org>/<name> format for the tokenizer

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

```bash
cargo run --release -- -i data/ -o output/ -t mlx-community/Llama-3.2-1B-Instruct-4bit
```

Postprocessing step:

```python
# currently the fields are hardcoded for training, but can be modified to suit the needs
# also assume the data has correct fields
import msgpack

def read_msgpack(filepath):
    with open(filepath, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return data

from datasets import Dataset
def gen_data(data):
    for row in data:
        yield {"input_ids": row[0], "labels": row[1], "position_ids": row[2],"length": row[3]}

data = read_msgpack("output/0.msgpack")
dataset = Dataset.from_generator(gen_data, {"data": data})
```

## Issues and caveats
- Only tokenizers with chat_template, bos_token, eos_token are supported

## Roadmap
[ ] Integrate with python directly with Maturin
[ ] Add more tests
[ ] Python reference code - For understanding