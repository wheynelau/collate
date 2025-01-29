use pyo3::IntoPyObject;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use rayon::prelude::*;

use crate::{binpacking, globals, template};

#[derive(Debug, Serialize, Deserialize)]
struct __Conversation {
    #[serde(alias = "conversations")]
    conversation: Vec<template::TextMessage>,
    
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
    conversation: Vec<String>,
}

#[derive(Clone, Serialize, IntoPyObject)]
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i32>,
    pub position_ids: Vec<u32>,
    pub length : u32,
}

impl TokenizedInput {
    pub fn merge(&mut self, other: &TokenizedInput) {
        // Change the method to take &mut self instead of &self
        self.input_ids.extend(other.input_ids.clone());
        self.labels.extend(other.labels.clone());
        self.position_ids.extend(other.position_ids.clone());
        self.length += other.length;
    }
}

pub fn read_jsonl(jsonl_path: &str, ct: template::ChatTemplate) -> Vec<String> {
    let jsonl = fs::read_to_string(jsonl_path).unwrap();
    let mut conversations: Vec<String> = Vec::new();
    for line in jsonl.lines() {
        let conv: __Conversation = serde_json::from_str(line).unwrap();
        let result = ct.apply(conv.conversation).unwrap();
        conversations.push(result);
    }
    conversations
}

fn get_msgpack_path(jsonl_path: &str, out_folder: String) -> String {
    let path = Path::new(jsonl_path);
    let file_stem = path.file_stem() // get the filename without extension
        .expect("Invalid file path")
        .to_str()
        .expect("Invalid file path");
    let mut out_path = Path::new(&out_folder).join(file_stem);
    out_path.set_extension("msgpack");
    out_path.to_str().unwrap().to_string()
}

fn tokenize_data(data:Vec<String>) -> Vec<TokenizedInput> {
    let mut inputs:Vec<TokenizedInput> = data
        .par_iter()
        .map(|conv| {
            let input_ids: Vec<u32> = globals::tokenize(conv).get_ids().to_owned();
            let mut labels: Vec<i32> = input_ids.iter().map(|x| *x as i32).collect();
            // replace labels.0 with -100
            labels[0] = -100;
            let position_ids = (0..input_ids.len() as u32).collect();
            let length = input_ids.len() as u32;
            TokenizedInput {
                input_ids,
                labels,
                position_ids,
                length,
            }
        })
        .collect();
    inputs.sort_by(|a, b| b.length.cmp(&a.length));
    inputs
}

pub fn python_process_jsonl(jsonl_path: &str, 
    template: template::ChatTemplate,
    out_folder: Option<String>,) -> Result<Vec<TokenizedInput>, std::io::Error> {
    
    // read jsonl for testing
    let data = read_jsonl(jsonl_path, template);
    // parallelize the tokenization
    let inputs: Vec<TokenizedInput> = tokenize_data(data);
    // bin packing
    let max_length = 8192;
    let bins = binpacking::create_bins(inputs, max_length);

    // serialize the bins
    let encoded = rmp_serde::to_vec(&bins).unwrap();
    if let Some(out_folder) = out_folder {

        let msg_pack_path = get_msgpack_path(jsonl_path, out_folder);
        fs::write(msg_pack_path, encoded)?;
    }

    Ok(bins)
}

pub fn single_jsonl_process(jsonl_path: &str, out_folder: &str,template: template::ChatTemplate) -> std::io::Result<()> {
    let msg_pack_path = get_msgpack_path(jsonl_path, out_folder.to_string());
    // read jsonl for testing
    let data = read_jsonl(jsonl_path, template);
    // parallelize the tokenization
    let inputs: Vec<TokenizedInput> = tokenize_data(data);
    // bin packing
    let max_length = 8192;
    let bins = binpacking::create_bins(inputs, max_length);

    // serialize the bins
    let encoded = rmp_serde::to_vec(&bins).unwrap();
    fs::write(msg_pack_path, encoded)?;
    globals::CURRENT_JSONL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    println!(
        "Processed {} out of {}",
        globals::CURRENT_JSONL.load(std::sync::atomic::Ordering::SeqCst),
        globals::TOTAL_JSONL.load(std::sync::atomic::Ordering::SeqCst)
    );

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
}