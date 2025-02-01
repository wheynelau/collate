use pyo3::IntoPyObject;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::{fs, path::Path};
use arrow::array::{ Int32Array, Int16Array };
use arrow::datatypes::{ Schema, Field, DataType };
use arrow::record_batch::RecordBatch;
use arrow::ipc:: writer::FileWriter;
use std::sync::Arc;

use rayon::prelude::*;
use indicatif::{ParallelProgressIterator,ProgressStyle,ProgressBar};

use crate::{binpacking, globals, template};

#[derive(Debug, Serialize, Deserialize)]
pub struct __Conversation {
    #[serde(alias = "conversations")]
    conversation: Vec<template::TextMessage>,
    
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
    conversation: Vec<String>,
}

#[derive(Clone, Serialize, IntoPyObject)]
pub struct TokenizedInput {
    pub input_ids: Vec<i32>, // use i32 for arrow
    pub labels: Vec<i32>,
    pub position_ids: Vec<i32>,
    pub length : i32,
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
    // Is this faster than using BufReader?
    let jsonl = fs::read_to_string(jsonl_path).unwrap();
    let length = jsonl.lines().count();
    println!("Number of lines: {}", length);
    let style = ProgressStyle::with_template("Serializing: [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Invalid progress style");
    let pb = ProgressBar::new(length as u64);
    pb.set_style(style);
    let conversations: Vec<String> = jsonl
        .par_lines()
        .progress_with(pb)
        .map(|item| {
            let conv: __Conversation = serde_json::from_str(item).unwrap();
            let result = ct.apply(conv.conversation).unwrap();
            result
        }).collect();
    conversations
}

// pub fn read_jsonl(jsonl_path: &str, ct: template::ChatTemplate) -> Vec<String> {
//     let file = File::open(jsonl_path).unwrap();
//     let mut reader = BufReader::new(file);
//     println!("Reading jsonl file: {}", jsonl_path);
//     // Count lines
//     let line_count = reader.lines().count();
//     println!("Number of lines: {}", line_count);
//     // Reset reader position to start
//     reader.seek(SeekFrom::Start(0)).unwrap();
    
//     // Pre-allocate vector with capacity
//     let mut conversations: Vec<String> = Vec::with_capacity(line_count);

//     let style = ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}");
    
//     for line in reader.lines().progress_with_style {
//         let line = line.unwrap();
//         let conv: Conversation = serde_json::from_str(&line).unwrap();
//         let result = ct.apply(conv.conversation).unwrap();
//         conversations.push(result);
//     }
//     conversations
// }
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

fn get_arrow_path(jsonl_path: &str, out_folder: String) -> String {
    let path = Path::new(jsonl_path);
    let file_stem = path.file_stem() // get the filename without extension
        .expect("Invalid file path")
        .to_str()
        .expect("Invalid file path");
    let mut out_path = Path::new(&out_folder).join(file_stem);
    out_path.set_extension("arrow");
    out_path.to_str().unwrap().to_string()
}

fn tokenize_data(data:Vec<String>) -> Vec<TokenizedInput> {
    let style = ProgressStyle::with_template("Tokenizing: [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Invalid progress style");
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(style);
    let mut inputs:Vec<TokenizedInput> = data
        .par_iter()
        .progress_with(pb)
        .map(|conv| {
            let input_ids: Vec<u32> = globals::tokenize(conv).get_ids().to_owned();
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
        })
        .collect();
    inputs.sort_by(|a, b| b.length.cmp(&a.length));
    inputs
}

pub fn python_process_jsonl(jsonl_path: &str, 
    template: template::ChatTemplate,
    max_length: i32,
    out_folder: Option<String>,
    ) -> Result<Vec<TokenizedInput>, std::io::Error> {
    
    // read jsonl for testing
    let data = read_jsonl(jsonl_path, template);
    // parallelize the tokenization
    let inputs: Vec<TokenizedInput> = tokenize_data(data);
    // bin packing
    
    println!("Creating bins");
    let bins = binpacking::create_bins(inputs, max_length);

    println!("Done!");
    // serialize the bins
    let encoded = rmp_serde::to_vec(&bins).unwrap();
    if let Some(out_folder) = out_folder {

        let msg_pack_path = get_msgpack_path(jsonl_path, out_folder);
        fs::write(msg_pack_path, encoded)?;
    }

    Ok(bins)
}

fn save_to_arrow(data: Vec<TokenizedInput>, path: &str) -> std::io::Result<()> {
    // let input_ids = data.iter().map(|x| x.input_ids.clone() as Vec<i32>).collect::<Vec<_>>();
    // let labels = data.iter().map(|x| x.labels.clone()).collect::<Vec<_>>();
    // let position_ids = data.iter().map(|x| x.position_ids.clone()).collect::<Vec<_>>();
    // let length = data.iter().map(|x| x.length).collect::<Vec<_>>();
    let schema = Schema::new(vec![
        Field::new("input_ids", DataType::Int32, false),
        Field::new("labels", DataType::Int32, false),
        Field::new("position_ids", DataType::Int32, false),
    ]);
    let path = Path::new(path);
    let mut buffer = File::create(path).expect("create file error");
    let mut writer = FileWriter::try_new(&mut buffer, &schema).expect("Error creating writer");
    for line in data.iter() {
        let input_ids = Int32Array::from(line.input_ids.clone());
        let labels = Int32Array::from(line.labels.clone());
        let position_ids = Int32Array::from(line.position_ids.clone());
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()), 
            vec![Arc::new(input_ids), 
            Arc::new(labels), 
            Arc::new(position_ids), 
            ])
            .expect("Error creating record batch");
        
        writer.write(&batch).expect("Error writing to file");
        }
        writer.finish().expect("Error finishing writing to file");
    Ok(())
}

pub fn single_jsonl_process(jsonl_path: &str, 
    out_folder: &str,
    template: template::ChatTemplate) -> std::io::Result<()> {
    let arrow_path = get_arrow_path(jsonl_path, out_folder.to_string());
    // read jsonl for testing
    let data = read_jsonl(jsonl_path, template);
    // parallelize the tokenization
    let inputs: Vec<TokenizedInput> = tokenize_data(data);
    // bin packing
    let max_length = 8192;
    let bins = binpacking::create_bins(inputs, max_length);

    // serialize the bins
    // let encoded = rmp_serde::to_vec(&bins).unwrap();
    // fs::write(msg_pack_path, encoded)?;
    save_to_arrow(bins, arrow_path.as_str())?;
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