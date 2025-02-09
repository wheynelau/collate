# collate-rs

Pet project on learning the data collation for huggingface [link](https://huggingface.co/blog/packing-with-FA2)

## Pre-requisites

- Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

This works for non root users as well.
- Other packages may be needed (Linux tested)
```bash
sudo apt install libssl-dev pkg-config build-essential
```

### Installing to current directory

```bash
cargo install --git <this-repo> --branch main
```

## Info

```python
# the below is a simplified version of the code in python
input = "data.jsonl"
data = read_jsonl(input)
heap = maxheap()
for line in data:
    conversation = tokenizer.apply_chat_template(line["conversation"])
    inputs_ids = conversation.input_ids
    labels = conversation.input_ids
    labels[0] = -100
    position_ids = list(range(len(inputs_ids)))
    # rust allows ordering via the length key
    heap.push({"input_ids": inputs_ids, "labels": labels, "position_ids": position_ids, "length": len(inputs_ids)})

curr_bin = {}
curr_len = 0
while not heap.is_empty():
    batch = heap.pop()
    if curr_len == 0:
        # only first iteration
        curr_bin = batch
        curr_len = batch["length"]
    elif curr_len + batch["length"] < max_len:
        curr_bin.merge(batch) # merge is a function to extend the lists
        curr_len += batch["length"]
    else:
        write_bin(curr_bin) # write bin writes to the output file , jsonl/arrow
        curr_bin = batch
        curr_len = batch["length"]
```
## Usage

Preprocessing step:

```bash
cargo run --release -- --help

This program reads a folder with jsonl files and packs them into the chosen format

Usage: collate [OPTIONS] --input <INPUT> --output <OUTPUT> --tokenizer <TOKENIZER>

Options:
  -i, --input <INPUT>
          Input to the root folder, should contain jsonl files like so - path/*.jsonl or just a single file

  -o, --output <OUTPUT>
          Output folder for the JSONL files, will write the jsonl as their own files
              in the output folder. Eg. input/file.jsonl -> output/file.msgpack

  -t, --tokenizer <TOKENIZER>
          Accepts huggingface <org>/<name> format for the tokenizer

  -m, --max-length <MAX_LENGTH>
          Max length of the tokenized input
          
          [default: 8192]

  -f, --format <FORMAT>
          Format of output file, [jsonl,arrow,msgpack]
          
          [default: arrow]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version

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
- The process reads the entire jsonl file into memory, to speed up the process. This results in a high memory overhead.

## Roadmap
[x] Integrate with python directly with Maturin - Completed but not very performant  
[ ] Add more tests  
[x] Python reference code - For understanding  
[ ] Reduce memory overhead  
