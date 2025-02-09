// Handles bin packing of TokenizedInput

use crate::conversations::TokenizedInput;
use arrow::array::builder::{GenericListBuilder, PrimitiveBuilder};
use arrow::array::types::Int32Type;
use arrow::array::ArrowPrimitiveType;
use arrow::array::ListArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

pub fn bin_and_save(
    mut inputs: BinaryHeap<TokenizedInput>,
    max_length: i32,
    arrow_path: String,
    pb: ProgressBar,
) {
    pb.reset();
    pb.set_length(inputs.len() as u64);
    let style = ProgressStyle::with_template("Preparing writer for {msg} [{elapsed_precise} / {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}")
        .expect("Invalid progress style");
    pb.set_style(style);
    pb.set_message(arrow_path.to_owned());
    let mut curr_length = 0;
    let mut curr_bin = TokenizedInput::new();

    // TODO: Use List instead of LargeList
    let schema = Schema::new(vec![
        Field::new(
            "input_ids",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
        Field::new(
            "position_ids",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
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
    record_vec.push(curr_bin);
    pb.reset();
    // start of writer
    let writer_style = ProgressStyle::with_template("Writing to {msg}: {spinner}")
        .expect("Invalid progress style");
    pb.set_style(writer_style);
    pb.enable_steady_tick(Duration::from_millis(100));
    // chunk the files
    // Reason for chunking, arrow offsets are limited to i32::MAX, unless using LargeList
    // To be experimented if this is faster than using LargeList
    let chunk_size = i32::MAX / max_length;
    let num_chunks = record_vec.len() as i32 / chunk_size + 1;
    record_vec
        .par_chunks(chunk_size as usize)
        .enumerate()
        .for_each(|(idx, chunk)| {
            // new name for arrow_path
            let arrow_path = arrow_path.clone();
            // easy way is replacing .arrow with {num}-of-{num_chunks}.arrow
            let arrow_path = arrow_path.replace(
                ".arrow",
                &format!("-{:04}-of-{:04}.arrow", idx + 1, num_chunks),
            );
            let mut buffer = File::create(&arrow_path).expect("create file error");
            let mut writer = StreamWriter::try_new_buffered(&mut buffer, &schema)
                .expect("Error creating writer");
            write_bin_to_writer(chunk, &mut writer, &schema);
        });
    // explicitly drop the writer to free memory
    pb.finish_and_clear();
}
fn from_iter_primitive_no_option<T, I>(iter: I) -> ListArray
where
    T: ArrowPrimitiveType,
    I: IntoIterator<Item = Vec<<T as ArrowPrimitiveType>::Native>>,
{
    let iter = iter.into_iter();
    let size_hint = iter.size_hint().0;
    let mut builder =
        GenericListBuilder::<i32, _>::with_capacity(PrimitiveBuilder::<T>::new(), size_hint);

    for inner_vec in iter {
        for value in inner_vec {
            builder.values().append_value(value); // Append non-optional values
        }
        builder.append(true); // Always append true since there are no missing values
    }
    builder.finish()
}

// TODO: Can we implement a streaming writer?
fn write_bin_to_writer<W>(bin: &[TokenizedInput], writer: &mut StreamWriter<W>, schema: &Schema)
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
    // explicitly drop the batch to free memory
    drop(batch);
}

pub fn bin_save_to_jsonl(
    mut inputs: BinaryHeap<TokenizedInput>,
    max_length: i32,
    jsonl_path: String,
    _pb: ProgressBar,
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
