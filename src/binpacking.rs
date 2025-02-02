// Handles bin packing of TokenizedInput

use crate::conversations::TokenizedInput;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
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
/// Takes in a max heap of TokenizedInput and bins them into a max_length
fn writer(
    rx: Receiver<TokenizedInput>,
    arrow_path: String,
    schema: Schema,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(&arrow_path);
    let mut buffer = File::create(path).expect("create file error");
    let mut writer =
        StreamWriter::try_new_buffered(&mut buffer, &schema).expect("Error creating writer");
    while let Ok(input) = rx.recv() {
        write_bin_to_writer(input, &mut writer, &schema);
    }
    writer.finish().expect("Error finishing writing to file");
    Ok(())
}
pub fn bin_and_save(mut inputs: BinaryHeap<TokenizedInput>, max_length: i32, arrow_path: String) {
    println!("Starting binning and saving to arrow");
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
    // let path = Path::new(&arrow_path);
    // let mut buffer = File::create(path).expect("create file error");
    // let mut writer = FileWriter::try_new_buffered(&mut buffer, &schema).expect("Error creating writer");
    // Can't parallelize this because we need to preserve order
    // Also due to heap
    let (tx, rx): (Sender<TokenizedInput>, Receiver<TokenizedInput>) = unbounded();
    std::thread::spawn(move || {
        writer(rx, arrow_path, schema.clone()).expect("Error writing to file");
    });
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
            tx.send(curr_bin).expect("Error sending to channel");
            curr_bin = input;
        }
    }
    tx.send(curr_bin).expect("Error sending to channel");
    drop(tx);
}
fn write_bin_to_writer<W>(bin: TokenizedInput, writer: &mut StreamWriter<W>, schema: &Schema)
where
    W: std::io::Write,
{
    let input_ids = Int32Array::from(bin.input_ids.clone());
    let labels = Int32Array::from(bin.labels.clone());
    let position_ids = Int32Array::from(bin.position_ids.clone());
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(input_ids),
            Arc::new(labels),
            Arc::new(position_ids),
        ],
    )
    .expect("Error creating record batch");

    writer.write(&batch).expect("Error writing to file");
}

pub fn bin_save_to_jsonl(
    mut inputs: BinaryHeap<TokenizedInput>,
    max_length: i32,
    jsonl_path: String,
) {
    let style = ProgressStyle::with_template("Writing: [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}")
    .expect("Invalid progress style");
    let pb = ProgressBar::new(inputs.len() as u64);
    pb.set_style(style);
    let mut curr_length = 0;
    let mut curr_bin = TokenizedInput::new();
    let file = File::create(Path::new(&jsonl_path)).expect("Create file error");
    let mut writer = BufWriter::new(file);
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
            write_bin_to_jsonl(curr_bin, &mut writer);
            curr_bin = input;
        }
    }
    write_bin_to_jsonl(curr_bin, &mut writer);
    println!("Finished writing to file");
    writer.flush().expect("Error finishing writing to file");
}

fn write_bin_to_jsonl(bin: TokenizedInput, writer: &mut BufWriter<File>) {
    let json = serde_json::to_string(&bin).expect("Error serializing to json");
    writeln!(writer, "{}", json).expect("Error writing to file");
}
