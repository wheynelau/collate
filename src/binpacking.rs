// Handles bin packing of TokenizedInput

use crate::{conversations::TokenizedInput, time_it};
use arrow::array::builder::{GenericListBuilder, PrimitiveBuilder};
use arrow::array::types::Int32Type;
use arrow::array::ArrowPrimitiveType;
use arrow::array::LargeListArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
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
// fn writer(
//     rx: Receiver<TokenizedInput>,
//     arrow_path: String,
//     schema: Schema,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let path = Path::new(&arrow_path);
//     let mut buffer = File::create(path).expect("create file error");
//     let mut writer =
//         StreamWriter::try_new_buffered(&mut buffer, &schema).expect("Error creating writer");
//     while let Ok(input) = rx.recv() {
//         write_bin_to_writer(input, &mut writer, &schema);
//     }
//     writer.finish().expect("Error finishing writing to file");
//     Ok(())
// }
pub fn bin_and_save(
    mut inputs: BinaryHeap<TokenizedInput>,
    max_length: i32,
    arrow_path: String,
    handles: &mut Vec<std::thread::JoinHandle<()>>,
) {
    println!("Starting binning and saving to arrow");
    let style = ProgressStyle::with_template("Writing: [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}")
    .expect("Invalid progress style");
    let pb = ProgressBar::new(inputs.len() as u64);
    pb.set_style(style);
    let mut curr_length = 0;
    let mut curr_bin = TokenizedInput::new();
    let schema = Schema::new(vec![
        Field::new(
            "input_ids",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
        Field::new(
            "labels",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
        Field::new(
            "position_ids",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
    ]);
    let mut record_vec = Vec::new();
    while let Some(mut input) = inputs.pop() {
        pb.inc(1);
        if input.length >= max_length {
            // add directly to record_vec
            input.truncate(max_length);
            record_vec.push(input);
            continue;
        }
        // always starts here
        if curr_length == 0 {
            curr_length = input.length;
            curr_bin.merge(&input);
        } else if curr_length + input.length <= max_length {
            curr_length += input.length;
            curr_bin.merge(&input);
        } else {
            curr_length = input.length;
            record_vec.push(curr_bin.clone());
            // tx.send(curr_bin).expect("Error sending to channel");
            curr_bin = input;
        }
    }

    pb.finish();
    let handle = std::thread::spawn(move || {
        let mut buffer = File::create(&arrow_path).expect("create file error");
        let mut writer =
            StreamWriter::try_new_buffered(&mut buffer, &schema).expect("Error creating writer");
        let msg = format!("Writing to {}", arrow_path);
        time_it!(
            msg,
            write_bin_to_writer(record_vec, &mut writer, &schema)
        );
    });
    handles.push(handle);
}
fn from_iter_primitive_no_option<T, I>(iter: I) -> LargeListArray
where
    T: ArrowPrimitiveType,
    I: IntoIterator<Item = Vec<<T as ArrowPrimitiveType>::Native>>,
{
    let iter = iter.into_iter();
    let size_hint = iter.size_hint().0;
    let mut builder =
        GenericListBuilder::<i64, _>::with_capacity(PrimitiveBuilder::<T>::new(), size_hint);

    for inner_vec in iter {
        for value in inner_vec {
            builder.values().append_value(value); // Append non-optional values
        }
        builder.append(true); // Always append true since there are no missing values
    }
    builder.finish()
}

// TODO: Can we implement a streaming writer?
// fn writer(
//     rx: Receiver<TokenizedInput>,
//     arrow_path: String,
//     schema: Schema,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let path = Path::new(&arrow_path);
//     let mut buffer = File::create(path).expect("create file error");
//     let mut writer =
//         StreamWriter::try_new_buffered(&mut buffer, &schema).expect("Error creating writer");
//     while let Ok(input) = rx.recv() {
//         write_bin_to_writer(input, &mut writer, &schema);
//     }
//     writer.finish().expect("Error finishing writing to file");
//     Ok(())
// }
fn write_bin_to_writer<W>(bin: Vec<TokenizedInput>, writer: &mut StreamWriter<W>, schema: &Schema)
where
    W: std::io::Write,
{
    let inputs_listarray =
        from_iter_primitive_no_option::<Int32Type, _>(bin.iter().map(|bin| bin.input_ids.clone()));
    let labels_listarray =
        from_iter_primitive_no_option::<Int32Type, _>(bin.iter().map(|bin| bin.labels.clone()));
    let positions_listarray = from_iter_primitive_no_option::<Int32Type, _>(
        bin.iter().map(|bin| bin.position_ids.clone()),
    );

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(inputs_listarray),
            Arc::new(labels_listarray),
            Arc::new(positions_listarray),
        ],
    )
    .expect("Error creating record batch");

    writer.write(&batch).expect("Error writing to file");
    writer.finish().expect("Error finishing writing to file");
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

    // Note: For now, we allow that there is possibility that the first entry is already
    // greater than the max_length
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
