// src/globals.rs
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use tokenizers;


/// Tokenizer object
///
/// This is a `OnceLock<tokenizers::Tokenizer>` that will be initialized when called with
/// `get_or_init` and a closure that returns a `tokenizers::Tokenizer`
///
/// # Example
///
/// ```
/// pub mod globals;
///
/// globals::TOKENIZER.get_or_init(|| {
///    tokenizers::Tokenizer::from_pretrained("openai-community/gpt2", None).unwrap()
/// });
///
/// ```
static TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();

pub static TOTAL_JSONL: AtomicU64 = AtomicU64::new(0);
pub static CURRENT_JSONL: AtomicU64 = AtomicU64::new(0);
/// Helper function to tokenize directly
///
/// This function will tokenize the content and return the encoding directly, abstracting the need to call `get` and `unwrap` on the OnceLock.
///
// # Arguments
///
/// * `content` : `&str` - The content to tokenize
///
/// # Returns
///
/// * `tokenizers::Encoding` - The tokenized content
///
/// # Example
///
/// ```
/// pub mod globals;
///
///
/// globals::init_tokenizer(&"openai-community/gpt2".to_string());
/// let content = "Hello world";
/// let encoding = globals::tokenize(content);
///
/// ```
///
/// # Panics
///
/// This function will panic if the tokenizer has not been initialized
pub fn tokenize(content: &str) -> tokenizers::Encoding {
    TOKENIZER
        .get()
        .expect("Tokenizer has not been initialized")
        .encode(content, false)
        .unwrap()
}

/// Helper function to initialize the tokenizer
///
/// This may be called at the beginning of the program if choosing to use a specific tokenizer
///
/// # Arguments
///
/// * `tokenizer_name` - `&String` - The name of the tokenizer to use, this should be in the format of `huggingface <org>/<name>` or a path to a tokenizer.json file
///
/// # Example
///
/// ```
/// pub mod globals;
///
/// globals::init_tokenizer(&"openai-community/gpt2".to_string());
///
/// // Continue with the program
pub fn init_tokenizer(tokenizer_name: &String) {
    if tokenizer_name.ends_with(".json") {
        println!("Loading tokenizer from file: {}", tokenizer_name);
        TOKENIZER
            .set(tokenizers::Tokenizer::from_file(tokenizer_name).unwrap())
            .expect("Unable to load tokenizer");
    } else {
        println!("Loading tokenizer: {}", tokenizer_name);
        TOKENIZER
            .set(tokenizers::Tokenizer::from_pretrained(tokenizer_name, None).unwrap())
            .expect("Unable to load tokenizer");
    }
}