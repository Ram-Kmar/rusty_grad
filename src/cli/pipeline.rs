use tysor::compiler::lexer::tokenize;

use crate::cli::args::CliOptions;
use crate::cli::output::print_section;

pub(crate) struct CompiledProgram {
    token_count: usize,
    program: Program,
    token_dump: Option<String>,
    ast_dump: Option<String>,
}

pub(crate) fn compile_source(
    source: &str,
    options: &CliOptions,
) -> Result<CompiledProgram, String> {
    let tokens = tokenize(source)?;
    let token_count = tokens.len();
    let token_dump = options.tokens.then(|| {
        tokens
            .iter()
            .map(|token| token.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    });

    Ok(CompiledProgram {
        token_count,
        program,
        token_dump,
        ast_dump,
    })
}

pub(crate) fn print_bootstrap_summary(options: &CliOptions, source_len: usize) {
    let input_path = options.input_path.as_ref().expect("validated input path");

    println!("tysor rust port bootstrap");
    println!("input={}", input_path.display());
    println!("bytes={source_len}");
    println!("backend={}", options.backend.as_str());
}

pub(crate) fn print_compile_summary(compiled: &CompiledProgram) {
    println!("tokens={}", compiled.token_count);
    println!(
        "program=configs:{} layers:{} functions:{} globals:{}",
        compiled.program.configs.len(),
        compiled.program.layers.len(),
        compiled.program.functions.len(),
        compiled.program.globals.len()
    );
}

pub(crate) fn print_requested_dumps(compiled: &CompiledProgram) {
    if let Some(token_dump) = &compiled.token_dump {
        print_section(
            "--- Tokenization Step ---",
            token_dump,
            "-------------------------",
        );
    }
    if let Some(ast_dump) = &compiled.ast_dump {
        print_section("--- Parsing Step ---", ast_dump, "--------------------");
    }
}
