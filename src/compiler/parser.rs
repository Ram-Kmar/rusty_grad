use std::cell::RefCell;

use crate::compiler::diagnositcs::Diagnostic;
use crate::compiler::lexer::{Token, TokenType, TokenValue};
use crate::compiler::source::SourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeBase{
    Unknown,
    Int,
    Bool,
    Float,
    Tensor,
    Void,
}

#[dervice(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub base: TypeBase,
    pub elements: Vec<Type>,
    pub callable_return: Option<Box<Type>>,
    pub scalar_dtype: Option<String>,
    pub tensor_dtype: Option<String>,
}

impl Default for Type {
    fn default() -> Self{
        Self::void()
    }

}

impl Type {
    pub fn unknown() -> Self {
        Self {
            base: TypeBase::Unknown,
            ..Self::void()
        }
    }

    pub fn int() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: None,
            ..Self::void()
        }
    }
    pub fn int16() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int16".to_string()),
            ..Self::void()
        }
    }
    pub fn int32() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int32".to_string()),
            ..Self::void()
        }
    }
    pub fn int64() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int64".to_string()),
            ..Self::void()
        }
    }
    pub fn float16() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float16".to_string()),
            ..Self::void()
        }
    }
    pub fn float32() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float32".to_string()),
            ..Self::void()
        }
    }
    pub fn float64() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float64".to_string()),
            ..Self::void()
        }
    }
    pub fn float() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: None,
            ..Self::void()
        }
    }
    pub fn bool() -> Self {
        Self {
            base: TypeBase::Bool,
            ..Self::void()
        }
    }
    pub fn void() -> Self {
        Self {
            base: TypeBase::Void,
            elements: Vec::new(),
            callable_return: None,
            scalar_dtype: None,
            tensor_dtype: None,
            tensor_shape_expr: None,
            tensor_rank: None,
        }
    }
    pub fn tensor(dtype: Option<String>, shape_expr: Option<String>, rank: Option<usize>) -> Self {
        Self {
            base: TypeBase::Tensor,
            tensor_dtype: dtype,
            tensor_shape_expr: shape_expr,
            tensor_rank: rank,
            ..Self::void()
        }
    }
    pub fn tuple(elements: Vec<Type>) -> Self {
        Self {
            base: TypeBase::Tuple,
            elements,
            ..Self::void()
        }
    }
    pub fn list(elements: Vec<Type>) -> Self {
        Self {
            base: TypeBase::List,
            elements,
            ..Self::void()
        }
    }
    pub fn callable(return_type: Type) -> Self {
        Self {
            base: TypeBase::Callable,
            callable_return: Some(Box::new(return_type)),
            ..Self::void()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallArgument{
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    IntLit(i32),
    IntLit64(i64),
    FloatLit(f32),
    FloatLit64(f64),
    BoolLit(bool),
    StringLit(String),
    Identifier(String),
    Call {
        callee: String,
        args: Vec<CallArgument>,
    },
    Repeat {
        stage: Box<Expr>,
        count: Box<Expr>,
    },
    Unary {
        operand: Box<Expr>,
        op: TokenType,
    },
    Binary {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: TokenType,
    },
    Ternary {
        then_expr: Box<Expr>,
        condition: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Arrow {
        source: Box<Expr>,
        stages: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub span: SourceSpan,
    pub kind: ExprKind,
}

impl Expr {
    fn span(&self) -> SourceSpan {
        self.span.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
    pub array_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignStmt {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfBranch {
    pub condition: Expr,
    pub body: Box<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Return(Expr),
    Expr(Expr),
    VarDecl(VarDecl),
    Assign(AssignStmt),
    Scope(Vec<Stmt>),
    If {
        condition: Expr,
        then_stmt: Box<Stmt>,
        elifs: Vec<IfBranch>,
        else_stmt: Option<Box<Stmt>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub span: SourceSpan,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub name: String,
    pub ty: Type,
    pub default_value: Option<Expr>,
    pub mutable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub span: SourceSpan,
    pub name: String,
    pub args: Vec<Arg>,
    pub return_type: Type,
    pub body: Stmt,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub functions: Vec<Function>,
    pub globals: Vec<Stmt>,
}

pub struct Parser{
    tokens: Vec<Token>,
    index: usize,
}

impl Parser{
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            index: 0,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut program = Program::default();
        while let Some(token) = self.peek(0).cloned() {
            match token.kind {
                TokenType::Newline | TokenType::Dedent => {
                    self.consume();
                }
                TokenType::Eof => break,
                TokenType::Fn => program.functions.push(self.parse_function()?),
                TokenType::Ident if self.peek_kind(1) == Some(TokenType::Eq) => {
                    program.globals.push(self.parse_stmt()?)
                }
                TokenType::Ident if self.peek_kind(1) == Some(TokenType::Colon) => {
                    return Err(self.error_token(
                        "Top-level variable declarations must start with 'let'",
                        &token,
                    ));
                }
                TokenType::Mut => {
                    return Err(
                        self.error_token("Mutable declarations must start with 'let mut'", &token)
                    );
                }
                _ => return Err(self.error_here("Unexpected token at top level")),
            }
        }
        Ok(program)
    }

    pub fn token_to_string(token: &Token) -> String {
        token.to_string()
    }

    fn parse_expression(&mut self) -> Result<Expr, String> {
        // Start with the lowest-precedence expression form and let each layer
        // delegate to the next tighter-binding form.
        self.parse_tuple_expression()
    }

    fn parse_tuple_expression(&mut self) -> Result<Expr, String> {
        let first = self.parse_conditional_expression()?;
        if self.peek_kind(0) == Some(TokenType::Comma) {
            let span = first.span();
            let mut elements = vec![first];
            while self.peek_kind(0) == Some(TokenType::Comma) {
                self.consume();
                elements.push(self.parse_conditional_expression()?);
            }
            Ok(Expr {
                span,
                kind: ExprKind::Tuple(elements),
            })
        } else {
            Ok(first)
        }
    }

    fn parse_conditional_expression(&mut self) -> Result<Expr, String> {
        let expr = self.parse_pipeline_expression()?;
        if self.peek_kind(0) == Some(TokenType::If) {
            let if_tok = self.consume();
            let condition = self.parse_pipeline_expression()?;
            self.expect(TokenType::Else, "Expected 'else' in ternary expression")?;
            let else_expr = self.parse_conditional_expression()?;
            Ok(Expr {
                span: span_of(&if_tok),
                kind: ExprKind::Ternary {
                    then_expr: Box::new(expr),
                    condition: Box::new(condition),
                    else_expr: Box::new(else_expr),
                },
            })
        } else {
            Ok(expr)
        }
    }

    fn parse_pipeline_expression(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_logical_or_expression()?;
        if self.peek_kind(0) != Some(TokenType::Arrow) {
            return Ok(lhs);
        }
        let span = self.peek(0).map(span_of).unwrap_or_default();
        let mut stages = Vec::new();
        while self.peek_kind(0) == Some(TokenType::Arrow) {
            let op = self.consume();
            let rhs = self.parse_logical_or_expression()?;
            if count_stage_sites(&rhs) != 1 {
                return Err(self.error_token(
                    "RHS of '->' must contain exactly one callable stage site. Use 'stage()[n]' for repetition.",
                    &op,
                ));
            }
            stages.push(rhs);
        }
        Ok(Expr {
            span,
            kind: ExprKind::Arrow {
                source: Box::new(lhs),
                stages,
            },
        })
    }
}