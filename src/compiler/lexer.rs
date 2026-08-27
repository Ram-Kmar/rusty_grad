use crate::compiler::diagnostics::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenType {
    Return,
    Int,
    Float,
    Bool,
    Tensor,
    List,
    True,
    False,
    #[default]
    Ident,
    Eof,
}

impl TokenType {
    pub fn as_str(self) -> &'static str{
        match self {
            TokenType::Return => "RETURN",
            TokenType::Int => "INT",
            TokenType::Float => "FLOAT",
            TokenType::Bool => "BOOL",
            TokenType::Tensor => "TENSOR",
            TokenType::Tuple => "TUPLE",
            TokenType::List => "LIST",
            TokenType::True => "TRUE",
            TokenType::False => "FALSE",
            TokenType::False => "def",
            TokenType::Ident => "Ident",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    Int(i32),
    Int64(i64),
    Float(f32),
    Float64(f64),
    String(String),
}

pub struct Token {
    pub kind: TokenType,
    pub value: Option<TokenValue>,
    pub line: usize,
    pub column: usize,
}

fn push_token(
    tokens: &mut Vec<Token>,
    kind: TokenType,
    value: Option<TokenValue>,
    line: usize,
    column: usize,
){
    token.push(Token {
        kind,
        value,
        line,
        column,
    });
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            Some(TokenValue::Int(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            Some(TokenValue::Float(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            Some(TokenValue::String(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            None => write!(f, "{} ({}:{})", self.kind.as_str(), self.line, self.column),
        }
    }
}

pub fn tokenize_with_diagnostic(source: &str) -> Result<Vec<Token>, Diagnostic>{
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut line = 1usize;
    let mut col = 1usize;
    let mut indent_stack = vec![0usize];
    let mut start_of_line = true;
    let mut paren_level = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        if start_of_line {
            let mut current_indent = 0usize;
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                if chars[i] == ' ' {
                    current_indent += 1;
                } else {
                    current_indent = (current_indent / 8 + 1) * 8;
                }
                i += 1;
            }

            if i < chars.len() && chars[i] != '\n' && chars[i] != '#' {
                let is_continuation = chars[i] == '-' && i + 1 < chars.len() && chars[i + 1] == '>';
                if paren_level == 0 && !is_continuation {
                    if current_indent > *indent_stack.last().unwrap() {
                        indent_stack.push(current_indent);
                        push_token(
                            &mut tokens,
                            TokenType::Indent,
                            None,
                            line,
                            current_indent + 1,
                        );
                    } else {
                        while current_indent < *indent_stack.last().unwrap() {
                            indent_stack.pop();
                            push_token(
                                &mut tokens,
                                TokenType::Dedent,
                                None,
                                line,
                                current_indent + 1,
                            );
                        }
                        if current_indent != *indent_stack.last().unwrap() {
                            return Err(lexer_error("Indentation error", line, current_indent + 1));
                        }
                    }
                }
            }

            col = current_indent + 1;
            start_of_line = false;
        }

        if i >= chars.len() {
            break;
        }

        let c = chars[i];
        let start_col = col;

        if c.is_ascii_alphabetic() {
            let mut buf = String::new();
            buf.push(c);
            while i + 1 < chars.len()
                && (chars[i + 1].is_ascii_alphanumeric() || chars[i + 1] == '_')
            {
                i += 1;
                col += 1;
                buf.push(chars[i]);
            }

            match buf.as_str() {
                "return" => push_token(&mut tokens, TokenType::Return, None, line, start_col),
                "int" => push_token(&mut tokens, TokenType::Int, None, line, start_col),
                "bool" => push_token(&mut tokens, TokenType::Bool, None, line, start_col),
                "float" => push_token(&mut tokens, TokenType::Float, None, line, start_col),
                "tensor" => push_token(&mut tokens, TokenType::Tensor, None, line, start_col),
                "tuple" => push_token(&mut tokens, TokenType::Tuple, None, line, start_col),
                "list" => push_token(&mut tokens, TokenType::List, None, line, start_col),
                "true" => push_token(&mut tokens, TokenType::True, None, line, start_col),
                "false" => push_token(&mut tokens, TokenType::False, None, line, start_col),
                "def" => push_token(&mut tokens, TokenType::Fn, None, line, start_col),
                "Ident" => push_token(&mut tokens, TokenType::Ident, None, line, start_col),
                _ => push_token(
                    &mut tokens,
                    TokenType::Ident,
                    Some(TokenValue::String(buf)),
                    line,
                    start_col,
                ),
            }
            i += 1;
            col += 1;
            continue;
        }

        if c == '"' {
            let mut buf = String::new();
            i += 1;
            col += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\n' {
                    return Err(lexer_error("Unterminated string literal", line, col));
                }
                buf.push(chars[i]);
                i += 1;
                col += 1;
            }
            if i >= chars.len() {
                return Err(lexer_error("Unterminated string literal", line, col));
            }
            push_token(
                &mut tokens,
                TokenType::StringLit,
                Some(TokenValue::String(buf)),
                line,
                start_col,
            );
            i += 1;
            col += 1;
            continue;
        }
        // This is section for finding numerical types - int , float
        if c.is_ascii_digit() {
            let mut buf = String::new();
            buf.push(c);
            while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                i += 1;
                col += 1;
                buf.push(chars[i]);
            }
            let mut is_float = false;
            // This is section for finding floats
            if i + 1 < chars.len() && chars[i + 1] == '.' {
                is_float = true;
                i += 1;
                col += 1;
                buf.push(chars[i]);
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
            }
            // This is section for finding floats number - "1.23E-5"
            if i + 1 < chars.len() && (chars[i + 1] == 'e' || chars[i + 1] == 'E') {
                is_float = true;
                i += 1;
                col += 1;
                buf.push(chars[i]);
                if i + 1 < chars.len() && (chars[i + 1] == '+' || chars[i + 1] == '-') {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
            }
            // This section just register that float value
            if is_float {
                let value = buf
                    .parse::<f64>() // Takes string and parse to f64
                    .map_err(|err| {
                        lexer_error(
                            format!("Invalid float literal '{buf}': {err}"),
                            line,
                            start_col,
                        )
                    })?;
                push_token(
                    &mut tokens,
                    TokenType::FloatLit,
                    Some(TokenValue::Float(value)),
                    line,
                    start_col,
                );
            } else {
                // This section just register that int value
                let value = buf.parse::<i64>().map_err(|err| {
                    lexer_error(
                        format!("Invalid int literal '{buf}': {err}"),
                        line,
                        start_col,
                    )
                })?;
                push_token(
                    &mut tokens,
                    TokenType::IntLit,
                    Some(TokenValue::Int(value)),
                    line,
                    start_col,
                );
            }
            i += 1;
            col += 1;
            continue;
        }
        match c {
            ';' => {
                push_token(&mut tokens, TokenType::Semi, None, line, col);
                i += 1;
                col += 1;
            }
            ':' => {
                push_token(&mut tokens, TokenType::Colon, None, line, col);
                i += 1;
                col += 1;
            }
            '.' => {
                push_token(&mut tokens, TokenType::Dot, None, line, col);
                i += 1;
                col += 1;
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::EqEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Eq, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::Neq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Bang, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    push_token(&mut tokens, TokenType::AmpAmp, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Amp, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    push_token(&mut tokens, TokenType::PipePipe, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Pipe, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::LtEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Lt, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::GtEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Gt, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '+' => {
                push_token(&mut tokens, TokenType::Plus, None, line, col);
                i += 1;
                col += 1;
            }
            '-' => {
                // Arrow operator
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    push_token(&mut tokens, TokenType::Arrow, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Minus, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '*' => {
                push_token(&mut tokens, TokenType::Star, None, line, col);
                i += 1;
                col += 1;
            }
            '/' => {
                if i + 1 < chars.len() && chars[i + 1] == '/' {
                    push_token(&mut tokens, TokenType::DoubleSlash, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Slash, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '#' => {
                // '#' starts a command/comment line. '//' is kept as an operator.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
            }
            '(' => {
                paren_level += 1;
                push_token(&mut tokens, TokenType::OpenParen, None, line, col);
                i += 1;
                col += 1;
            }
            ')' => {
                if paren_level > 0 {
                    paren_level -= 1;
                }
                push_token(&mut tokens, TokenType::CloseParen, None, line, col);
                i += 1;
                col += 1;
            }
            '[' => {
                paren_level += 1;
                push_token(&mut tokens, TokenType::OpenBracket, None, line, col);
                i += 1;
                col += 1;
            }
            ']' => {
                if paren_level > 0 {
                    paren_level -= 1;
                }
                push_token(&mut tokens, TokenType::CloseBracket, None, line, col);
                i += 1;
                col += 1;
            }
            ',' => {
                push_token(&mut tokens, TokenType::Comma, None, line, col);
                i += 1;
                col += 1;
            }
            _ if c.is_whitespace() => {
                if c == '\n' {
                    if paren_level == 0 {
                        let mut next_is_continuation = false;
                        let mut j = i + 1;
                        while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                            j += 1;
                        }
                        if j + 1 < chars.len() && chars[j] == '-' && chars[j + 1] == '>' {
                            next_is_continuation = true;
                        }
                        if !next_is_continuation {
                            push_token(&mut tokens, TokenType::Newline, None, line, col);
                        }
                    }
                    line += 1;
                    col = 1;
                    start_of_line = true;
                } else {
                    col += 1;
                }
                i += 1;
            }
            _ => return Err(lexer_error(format!("Unknown character '{c}'"), line, col)),
        }
    }

    while indent_stack.len() > 1 {
        indent_stack.pop();
        push_token(&mut tokens, TokenType::Dedent, None, line, col);
    }
    push_token(&mut tokens, TokenType::Eof, None, line, col);
    Ok(tokens)
}

fn lexer_error(message: impl Into<String>, line: usize, column: usize) -> Diagnostic {
    Diagnostic::error("lexer", "L0001", message, line, column)
}
