use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct SourceSpan{
    pub line: usize,
    pub column: usize,
}

impl SourceSpan {
    pub fn new(line: size, column: usize) -> Self {
        Self { line, column }
    }
}

