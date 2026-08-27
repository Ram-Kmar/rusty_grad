use std::collections::BTreeMap;
use std::path::PathBuf; // PathBuf is a growable, owned file path.

use tysor::backend::core::kind::BackendKind;

#[derive(Debug, Default)]
pub struct CliOptions{
    pub input_path: Option<PathBuf>,
    pub tokens: bool,
    pub backend: BackendKind,
}

pub fn usage() -> &'static str {
    "tysor <input.ty> [options]
tysor debug-visualize [--api-port <port>] [--ui-port <port>] [--no-install]

Options:
  --tokens                      Print lexer tokens
  --ast                         Print parser AST
  --backend <local|metal|pytorch|cuda|rocm>
                                Execution backend, defaults to local
  -h, --help                    Show this help
  -V, --version                 Show version

Examples:
  tysor examples/tensor_tuple/simple_tensor.ty --tokens --ast"
}

//This function is setting the CliOptions struct.
pub fn parse_cli(raw_args: impl IntoIterator<Item = String>) -> Result<CliOptions, String>{
    let mut args = raw_args.into_iter();
    let mut options = CliOptions {
        backend: BackendKind::Cpu,
        ..CliOptions::default()
    };

    while let Some(arg) = args.next(){
        match arg.as_str() {
            "--tokens" => options.tokens = true,
            "--ast" => options.ast = true,
            "--backend" => {
                let backend = args
                    .next()
                    .ok_or_else(|| "missing value for --backend".to_string())?;
                options.backend = backend.parse()?;
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option: {arg}")),
            _ => {
                if options.input_path.is_some() {
                    return Err("multiple input paths provided".to_string());
                }
                options.input_path = Some(PathBuf::from(arg));
            }
        }
    }

    if options.input_path.is_none() {
        return Err("missing input path".to_string());
    }

    Ok(options)
}
