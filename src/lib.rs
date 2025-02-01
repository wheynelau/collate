use pyo3::prelude::*;

// lib.rs
pub mod args;
pub mod globals;
pub mod template;
pub mod conversations;
pub mod binpacking;
pub mod config;

#[pyfunction]
#[pyo3(signature = (input, tokenizer, max_length, out_folder=None))]
fn collate_jsonl(input: String,
    tokenizer: String,
    max_length:i32,
    out_folder: Option<String>) -> PyResult<Vec<conversations::TokenizedInput>> {
    globals::init_tokenizer(&tokenizer);
    // read config

    let config: config::TokenizerConfig = config::read_config(&tokenizer).unwrap();
    let template = template::ChatTemplate::new(
        config.chat_template,
        Some("<|begin_of_text|>".to_string()),
        Some("<|eot_id|>".to_string()),
    );
    // only read one jsonl
    let result = conversations::python_process_jsonl(
        &input,
        template,
        max_length,
        out_folder
    )?;

    Ok(result)
}

#[pymodule]
fn collate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collate_jsonl, m)?)
}
