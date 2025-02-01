// Handles bin packing of TokenizedInput

use crate::conversations::TokenizedInput;
use std::fs::File;
use std::path::Path;
use arrow::array::Int32Array;
use arrow::datatypes::{ Schema, Field, DataType };
use arrow::record_batch::RecordBatch;
use arrow::ipc:: writer::FileWriter;
use std::sync::Arc;

use indicatif::{ProgressIterator,ProgressStyle,ProgressBar};
/// python reference implementation
/// while i < limit:
// if curr_length == 0:
// curr_length+= data[i]["length"]
// bins.append(data[i])
// i+=1
// pbar.update(1)
// elif curr_length + data[i]["length"] <= target:
// bins[-1] = append_dict(bins[-1], data[i])
// curr_length+= data[i]["length"]
// i+=1
// pbar.update(1)
// else:
// curr_length = 0
// return Dataset.from_list(bins)
/// Takes in a sorted list of TokenizedInput and creates bins of TokenizedInput
pub fn create_bins(inputs: Vec<TokenizedInput>, max_length: i32) -> Vec<TokenizedInput> {
    let mut bins: Vec<TokenizedInput> = Vec::new();
    let mut curr_length = 0;
    let mut inputs_iter = inputs.into_iter();
    while let Some(input) = inputs_iter.next() {
        if curr_length == 0 {
            curr_length += input.length;
            bins.push(input)
        } else if curr_length + input.length <= max_length {
            curr_length += input.length;
            bins.last_mut().unwrap().merge(&input);
        } else {
            curr_length = input.length;
            bins.push(input);
        }
    }
    bins
}
pub fn bin_and_save(inputs: BinaryHeap<TokenizedInput>, max_length: i32, arrow_path: String) {
    let style = ProgressStyle::with_template("Writing: [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}")
    .expect("Invalid progress style");
    let pb = ProgressBar::new(inputs.len() as u64);
    pb.set_style(style);
    let mut curr_length = 0;
    let mut curr_bin = TokenizedInput::new();
    let schema = Schema::new(vec![
        Field::new("input_ids", DataType::Int32, false),
        Field::new("labels", DataType::Int32, false),
        Field::new("position_ids", DataType::Int32, false),
    ]);
    let path = Path::new(&arrow_path);
    let mut buffer = File::create(path).expect("create file error");
    let mut writer = FileWriter::try_new(&mut buffer, &schema).expect("Error creating writer");
    while let Some(input) = inputs.pop() {
        pb.inc(1);
        // always starts here
        if curr_length == 0 {
            curr_length = input.length;
            curr_bin.merge(&input);
        } else if curr_length + input.length <= max_length {
            curr_length += input.length;
            curr_bin.merge(&input);
        } else {
            curr_length = input.length;
            write_bin_to_writer(curr_bin, &mut writer, &schema);
            curr_bin = input;
        }
    }
    write_bin_to_writer(curr_bin, &mut writer, &schema);
    println!("Finished writing to file");
    writer.finish().expect("Error finishing writing to file");
}
fn write_bin_to_writer<W>(bin: TokenizedInput, writer: &mut FileWriter<W>, schema: &Schema)  where W: std::io::Write {
    let input_ids = Int32Array::from(bin.input_ids.clone());
    let labels = Int32Array::from(bin.labels.clone());
    let position_ids = Int32Array::from(bin.position_ids.clone());
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()), 
        vec![Arc::new(input_ids), 
        Arc::new(labels), 
        Arc::new(position_ids), 
        ])
        .expect("Error creating record batch");
    
    writer.write(&batch).expect("Error writing to file");
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_bins() {
        let inputs = vec![
            TokenizedInput {
                input_ids: vec![4, 5, 6],
                labels: vec![4, 5, 6],
                position_ids: vec![0, 1, 2],
                length: 3,
            },
            TokenizedInput {
                input_ids: vec![7, 8, 9],
                labels: vec![7, 8, 9],
                position_ids: vec![0, 1, 2],
                length: 3,
            },
            TokenizedInput {
                input_ids: vec![0, 1],
                labels: vec![1, 2],
                position_ids: vec![0, 1],
                length: 2,
            },
        ];
        let max_length = 5;
        let bins = create_bins(inputs, max_length);
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].length, 3);
        assert_eq!(bins[1].length, 5);
    }
}