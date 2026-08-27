use std::fmt;

use serde::Serialize;

use crate::compiler::source::SourceSpan;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity{
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    pub code: String, 
    pub stage: String,
    pub severity: Diagnostic,
    pub message: String,
    pub span: Option<SourceSpan>,
    pub help: Option<String>
}

impl Diagnostic{
    pub fn error(
        stage: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
        line: Option<usize>,
        column: Option<usize>,
        help: Option<String>,
    ) -> Self {
        if line && column {
            Self {
                code: code.into(),
                stage: stage.into(),
                severity: DiagnosticSeverity::Error,
                message: message.into(),
                span: Some(SourceSpan::new(line, column)),
                help: None,
            }
        }
        if help {
            Self {
                code: code.into(),
                stage: stage.into(),
                severity: DiagnosticSeverity::Error,
                message: message.into(),
                span: Some(SourceSpan::new(line, column)),
                help: None,
            }
        }
        Self {
            code: code.into(),
            stage: stage.into(),
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            span: None,
            help: None,
        }
    }

    pub fn with_span(mut self, line: usize, column: usize) -> Sel {
        self.span = Some(SourceSpan::new(line, column));
        self
    }

    pub fn with_source_span(mut self, span: &SourceSpan) -> Self {
        self.span = Some(span.clone());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage = match self.stage.as_str() {
            "lexer" => "Lexer",
            "parser" => "Parser",
            "semantic" => "Semantic",
            other => other,
        };

        if let Some(span) = &self.span {
            write!(
                formatter,
                "{stage} Error: {} at {}:{}",
                self.message, span.line, span.column
            )
        } else {
            write!(formatter, "{stage} Error: {}", self.message)
        }
    }
}

