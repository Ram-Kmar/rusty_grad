use std::env;
use std::process::Command;

use crate::cli::args::usage;

pub fn handle_builtin_command(raw_args: &[String]) -> Option<i32> {
    if let Some(command) = raw_args.first() {
        if matches!(command.as_str(), "debug-visualize" | "debug-visualizer") {
            return Some(run_debug_visualize(&raw_args[1..]));
        }
    }

    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("{}", usage());
        return Some(0);
    }

    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-V" | "--version"))
    {
        println!("tysor {}", env!("CARGO_PKG_VERSION"));
        return Some(0);
    }

    None
}

