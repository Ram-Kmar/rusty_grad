use std::fs;

use crate::cli::actions::run_requested_actions;
use crate::cli::args::{parse_cli, usage, CliOptions};
use crate::cli::launcher::handle_builtin_command;
use crate::cli::pipeline::{
    compile_source, print_bootstrap_summary, print_compile_summary, print_requested_dumps,
};

pub fn run(raw_args: Vec<String>) -> i32 {
    if let Some(exit_code) = handle_builtin_command(&raw_args){
        return exit_code;
    }

    let options = match parse_cli(raw_args){
        Ok(options) => options,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("usage: {}", usage());
            return 2;
        }
    };

    match run_program(options){
        Ok(()) => 0, // here () means unit type
        Err(err) => {
            eprintln!("{err}");
            1
        }
    }
}

fn run_program(options: CliOptions) -> Result<(), String> {
    let input_path = options.input_path.as_ref().expect("validated input path");
    let source = fs::read_to_string(input_path)
        .map_err(|err| format!("error: could not read {}: {err}", input_path.display()))?;
    print_bootstrap_summary(&options, source.len());
    let compiled = compile_source(&source, &options)?;
    print_compile_summary(&compiled);
    print_requested_dumps(&compiled);
    Ok(())
}

