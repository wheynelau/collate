use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufRead, BufReader};
use std::{fs, path::Path};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::{binpacking, globals, template};

#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(alias = "conversations")]
    conversation: Vec<template::TextMessage>,
}

#[derive(Clone, Serialize)]
pub struct TokenizedInput {
    pub input_ids: Vec<i32>, // use i32 for arrow
    pub labels: Vec<i32>,
    pub position_ids: Vec<i32>,
    pub length: i32,
}

impl Ord for TokenizedInput {
    fn cmp(&self, other: &Self) -> Ordering {
        self.length.cmp(&other.length)
    }
}

impl PartialOrd for TokenizedInput {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TokenizedInput {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
    }
}

impl Eq for TokenizedInput {}

impl Default for TokenizedInput {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizedInput {
    pub fn new() -> Self {
        TokenizedInput {
            input_ids: Vec::new(),
            labels: Vec::new(),
            position_ids: Vec::new(),
            length: 0,
        }
    }
    pub fn merge(&mut self, other: &TokenizedInput) {
        // Change the method to take &mut self instead of &self
        self.input_ids.extend(other.input_ids.clone());
        self.labels.extend(other.labels.clone());
        self.position_ids.extend(other.position_ids.clone());
        self.length += other.length;
    }
    pub fn truncate(&mut self, max_length: i32) {
        self.input_ids.truncate(max_length as usize);
        self.labels.truncate(max_length as usize);
        self.position_ids.truncate(max_length as usize);
        self.length = self.input_ids.len() as i32;
    }
}
// This function is used in the read_jsonl function
// It is used to parse the jsonl file and tokenize the conversation
fn parse_and_tokenize(item: &str, ct: template::ChatTemplate) -> TokenizedInput {
    let conv: Conversation = serde_json::from_str(item).unwrap();
    let result = ct.apply(conv.conversation).unwrap();
    let input_ids: Vec<u32> = globals::tokenize(&result).get_ids().to_owned();
    let input_ids: Vec<i32> = input_ids.iter().map(|x| *x as i32).collect();
    let mut labels: Vec<i32> = input_ids.clone();
    // replace labels.0 with -100
    labels[0] = -100;
    let position_ids = (0..input_ids.len() as i32).collect();
    let length = input_ids.len() as i32;
    TokenizedInput {
        input_ids,
        labels,
        position_ids,
        length,
    }
}

fn tokenize_jsonl(
    jsonl_path: &str,
    ct: template::ChatTemplate,
    pb: ProgressBar,
) -> BinaryHeap<TokenizedInput> {
    // Start of reading
    let reader_style = ProgressStyle::with_template(
        "Reading {msg} \
        [{elapsed_precise} / {eta_precise}] \
        {bar:40.cyan/blue} \
        {decimal_bytes:>7}/{decimal_total_bytes:7} \
        {decimal_bytes_per_sec}",
    )
    .expect("Invalid progress style");
    let file = fs::File::open(jsonl_path).expect("File not found");
    let file_size = file.metadata().expect("Error reading metadata").len();
    pb.set_length(file_size);
    pb.set_style(reader_style);
    pb.set_message(jsonl_path.to_owned());
    let reader = BufReader::new(file);
    let jsonl: Vec<String> = pb.wrap_read(reader).lines().map(|l| l.unwrap()).collect();
    pb.reset();

    // Start of tokenization
    let length = jsonl.len();
    let style = ProgressStyle::with_template(
        "Tokenizing {msg} \
         [{elapsed_precise} / {eta_precise}] \
         {bar:40.cyan/blue} \
         {pos:>7}/{len:7} \
         {per_sec}",
    )
    .expect("Invalid progress style");
    pb.set_length(length as u64);
    pb.set_style(style);
    pb.set_message(jsonl_path.to_owned());

    // Main loop, parse, tokenize and create a local heap
    let results: Vec<_> = jsonl
        .par_chunks(50_000)
        .map(|slice| {
            // Create a local heap for this chunk
            let mut local_heap = BinaryHeap::with_capacity(slice.len());

            slice.iter().for_each(|item| {
                pb.inc(1);
                let input: TokenizedInput = parse_and_tokenize(item, ct.clone());
                local_heap.push(input);
            });

            local_heap
        })
        .collect();
    // merge all the local heaps into one heap
    results
        .into_iter()
        .fold(BinaryHeap::with_capacity(length), |mut acc, mut x| {
            acc.append(&mut x);
            acc
        })
}

fn get_jsonl_path(jsonl_path: String, out_folder: String) -> String {
    let path = Path::new(&jsonl_path);
    let file_stem = path
        .file_stem() // get the filename without extension
        .expect("Invalid file path")
        .to_str()
        .expect("Invalid file path");
    let mut out_path = Path::new(&out_folder).join(file_stem);
    out_path.set_extension("jsonl");
    out_path.to_str().unwrap().to_string()
}

fn get_arrow_path(jsonl_path: String, out_folder: String) -> String {
    let path = Path::new(&jsonl_path);
    let file_stem = path
        .file_stem() // get the filename without extension
        .expect("Invalid file path")
        .to_str()
        .expect("Invalid file path");
    let mut out_path = Path::new(&out_folder).join(file_stem);
    out_path.set_extension("arrow");
    out_path.to_str().unwrap().to_string()
}

pub fn single_jsonl_process(
    jsonl_path: String,
    max_length: i32,
    out_folder: String,
    template: template::ChatTemplate,
    format: String,
    pb: MultiProgress,
) -> std::io::Result<()> {
    // read and tokenize in parallel
    let pb = pb.add(ProgressBar::new(0));
    let inputs: BinaryHeap<TokenizedInput> = tokenize_jsonl(&jsonl_path, template, pb.clone());

    // Dispatch the job to a thread because its not parallelisable and IO bound
    match format.to_ascii_lowercase().as_str() {
        "arrow" => {
            let arrow_path = get_arrow_path(jsonl_path, out_folder.to_string());
            binpacking::bin_and_save(inputs, max_length, arrow_path, pb);
        }
        "jsonl" => {
            let jsonl_path = get_jsonl_path(jsonl_path, out_folder.to_string());
            binpacking::bin_save_to_jsonl(inputs, max_length, jsonl_path, pb);
        }
        _ => {
            let _ = Err::<(), anyhow::Error>(anyhow::anyhow!("Format not supported"));
        }
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        let mut left = TokenizedInput {
            input_ids: vec![1, 2, 3],
            labels: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            length: 3,
        };
        let right = TokenizedInput {
            input_ids: vec![4, 5, 6],
            labels: vec![4, 5, 6],
            position_ids: vec![0, 1, 2],
            length: 3,
        };
        left.merge(&right);
        assert_eq!(left.input_ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(left.labels, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(left.position_ids, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(left.length, 6);
    }
    #[test]
    fn test_truncate() {
        let mut input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            labels: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            length: 3,
        };
        input.truncate(2);
        assert_eq!(input.input_ids, vec![1, 2]);
        assert_eq!(input.labels, vec![1, 2]);
        assert_eq!(input.position_ids, vec![0, 1]);
        assert_eq!(input.length, 2);
    }

    #[test]
    fn heap_sort() {
        let mut heap: BinaryHeap<TokenizedInput> = BinaryHeap::new();
        heap.push(TokenizedInput {
            input_ids: vec![1, 2, 3],
            labels: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            length: 1,
        });
        heap.push(TokenizedInput {
            input_ids: vec![4, 5, 6],
            labels: vec![4, 5, 6],
            position_ids: vec![0, 1, 2],
            length: 5,
        });
        heap.push(TokenizedInput {
            input_ids: vec![7, 8, 9],
            labels: vec![7, 8, 9],
            position_ids: vec![0, 1, 2],
            length: 2,
        });
        let mut sorted: Vec<TokenizedInput> = Vec::new();
        while let Some(item) = heap.pop() {
            sorted.push(item);
        }
        assert_eq!(sorted[0].length, 5);
        assert_eq!(sorted[1].length, 2);
        assert_eq!(sorted[2].length, 1);
    }
}
