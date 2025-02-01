#[macro_export]
macro_rules! time_it {
    ($task:expr, $code:expr) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();
        println!("{}: {:?}", $task, duration);
        result
    }};
}
