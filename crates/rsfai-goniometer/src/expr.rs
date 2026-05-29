//! A small f64 expression evaluator for the goniometer transformation formulas,
//! reproducing `numexpr`'s scalar evaluation bit-for-bit.
//!
//! pyFAI's `GeometryTransformation` (`goniometer.py`) feeds each PONI component's
//! formula string to `numexpr.NumExpr`, then calls it with the parameter / motor
//! values bound by name. For the scalar inputs a goniometer config uses, numexpr
//! evaluates in IEEE-754 f64: the four arithmetic ops `+ - * /`, unary minus, the
//! transcendentals `sin/cos/tan/sqrt/abs`, the power operator `**`, parenthesised
//! grouping, the constant `pi`, plus user-named variables and numeric literals.
//!
//! Bit-exactness rests on two verified facts (probed against the daq numexpr
//! build, see `golden/gen_golden_goniometer.py`):
//!   * numexpr's `sin/cos/tan/sqrt` and `+ - * /` on f64 scalars match the system
//!     libm Rust links — identical bits.
//!   * numexpr's `**` lowers an **integer** exponent to exponentiation-by-squaring
//!     (matching Rust's `f64::powi` bit-for-bit, including the `a**6` case where a
//!     left-fold multiply chain would differ by 1 ULP) and a **non-integer**
//!     exponent to `pow` (matching `f64::powf`). This evaluator applies exactly
//!     that split, so `**` is a structural match, not a coincidence of the
//!     formulas chosen.
//!
//! Parsing is recursive descent over the standard precedence ladder
//! (`+ -` < `* /` < unary `-` < `**` < atom), with `**` right-associative as in
//! Python/numexpr. Evaluation is a single pass over the parsed AST; the AST is
//! built once at construction (mirroring numexpr compiling the formula once into
//! `NumExpr`) and evaluated per call with a fresh variable binding.

use std::collections::BTreeMap;
use std::f64::consts::PI;

/// A parse / evaluation failure in a transformation formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprError {
    /// The formula text could not be parsed (with a human-readable reason).
    Parse(String),
    /// A variable referenced by the formula was not bound at evaluation time.
    UnboundVariable(String),
}

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprError::Parse(m) => write!(f, "formula parse error: {m}"),
            ExprError::UnboundVariable(v) => write!(f, "unbound variable in formula: {v}"),
        }
    }
}

impl std::error::Error for ExprError {}

/// The single-argument transcendental functions numexpr exposes that a
/// goniometer formula uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Func {
    Sin,
    Cos,
    Tan,
    Sqrt,
    Abs,
}

/// A parsed formula node. `pi` is parsed into a [`Ast::Const`] so it costs no
/// per-call lookup (numexpr binds `pi` as a constant too).
#[derive(Debug, Clone, PartialEq)]
enum Ast {
    Const(f64),
    Var(String),
    Neg(Box<Ast>),
    Add(Box<Ast>, Box<Ast>),
    Sub(Box<Ast>, Box<Ast>),
    Mul(Box<Ast>, Box<Ast>),
    Div(Box<Ast>, Box<Ast>),
    Pow(Box<Ast>, Box<Ast>),
    Call(Func, Box<Ast>),
}

/// A compiled transformation formula: the parsed AST plus the sorted set of
/// free-variable names it references (so a caller can pre-validate the binding).
#[derive(Debug, Clone)]
pub struct Formula {
    source: String,
    ast: Ast,
    variables: Vec<String>,
}

impl Formula {
    /// Parse a formula string. Returns a [`ExprError::Parse`] on malformed input.
    /// `pi` is recognised as the f64 constant `std::f64::consts::PI` (numexpr binds
    /// `numpy.pi`, the same f64 value).
    pub fn parse(source: &str) -> Result<Formula, ExprError> {
        let tokens = lex(source)?;
        let mut parser = Parser { tokens, pos: 0 };
        let ast = parser.parse_expr()?;
        if parser.pos != parser.tokens.len() {
            return Err(ExprError::Parse(format!(
                "unexpected trailing tokens in `{source}`"
            )));
        }
        let mut vars: Vec<String> = Vec::new();
        collect_vars(&ast, &mut vars);
        vars.sort();
        vars.dedup();
        Ok(Formula {
            source: source.to_string(),
            ast,
            variables: vars,
        })
    }

    /// The original formula text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The sorted, de-duplicated free-variable names this formula references
    /// (`pi` is a constant, never listed).
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Evaluate the formula with `vars` bound to the named values. An unbound
    /// reference is an [`ExprError::UnboundVariable`]. The arithmetic is the
    /// bit-exact f64 path described in the module docs.
    pub fn eval(&self, vars: &BTreeMap<String, f64>) -> Result<f64, ExprError> {
        eval_ast(&self.ast, vars)
    }
}

/// Evaluate a parsed AST against a variable binding, in IEEE-754 f64 with no FMA
/// contraction (the profile disables it) — the order of operations is fixed by
/// the parse tree, so the result is reproducible.
fn eval_ast(ast: &Ast, vars: &BTreeMap<String, f64>) -> Result<f64, ExprError> {
    Ok(match ast {
        Ast::Const(c) => *c,
        Ast::Var(name) => *vars
            .get(name)
            .ok_or_else(|| ExprError::UnboundVariable(name.clone()))?,
        Ast::Neg(a) => -eval_ast(a, vars)?,
        Ast::Add(a, b) => eval_ast(a, vars)? + eval_ast(b, vars)?,
        Ast::Sub(a, b) => eval_ast(a, vars)? - eval_ast(b, vars)?,
        Ast::Mul(a, b) => eval_ast(a, vars)? * eval_ast(b, vars)?,
        Ast::Div(a, b) => eval_ast(a, vars)? / eval_ast(b, vars)?,
        Ast::Pow(a, b) => {
            let base = eval_ast(a, vars)?;
            let exp = eval_ast(b, vars)?;
            pow(base, exp)
        }
        Ast::Call(func, a) => {
            let x = eval_ast(a, vars)?;
            match func {
                Func::Sin => x.sin(),
                Func::Cos => x.cos(),
                Func::Tan => x.tan(),
                Func::Sqrt => x.sqrt(),
                Func::Abs => x.abs(),
            }
        }
    })
}

/// The `**` operator, matching numexpr's lowering: an exponent that is exactly an
/// integer (and fits `i32`) goes through `f64::powi` (exponentiation-by-squaring,
/// bit-identical to numexpr's integer-power expansion); anything else through
/// `f64::powf`. The integer test is on the *value*, so `2.0` and `2` take the same
/// `powi` branch numexpr does.
fn pow(base: f64, exp: f64) -> f64 {
    if exp.is_finite() && exp.fract() == 0.0 && exp.abs() <= i32::MAX as f64 {
        base.powi(exp as i32)
    } else {
        base.powf(exp)
    }
}

/// Collect free-variable names referenced by an AST.
fn collect_vars(ast: &Ast, out: &mut Vec<String>) {
    match ast {
        Ast::Const(_) => {}
        Ast::Var(name) => out.push(name.clone()),
        Ast::Neg(a) | Ast::Call(_, a) => collect_vars(a, out),
        Ast::Add(a, b) | Ast::Sub(a, b) | Ast::Mul(a, b) | Ast::Div(a, b) | Ast::Pow(a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
    }
}

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Pow,
    LParen,
    RParen,
}

/// Tokenise a formula string. Whitespace is ignored; `**` is recognised before
/// `*`. Numeric literals are parsed with Rust's f64 `FromStr`, which matches
/// Python's float literal parsing for the decimal / scientific forms a formula
/// uses.
fn lex(src: &str) -> Result<Vec<Tok>, ExprError> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut toks = Vec::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                if i + 1 < bytes.len() && bytes[i + 1] as char == '*' {
                    toks.push(Tok::Pow);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                // Consume a numeric literal: digits, one decimal point, and an
                // exponent `e`/`E` with optional sign.
                while i < bytes.len() {
                    let ch = bytes[i] as char;
                    if ch.is_ascii_digit() || ch == '.' {
                        i += 1;
                    } else if ch == 'e' || ch == 'E' {
                        i += 1;
                        if i < bytes.len() && (bytes[i] as char == '+' || bytes[i] as char == '-') {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                let text = &src[start..i];
                let value = text
                    .parse::<f64>()
                    .map_err(|_| ExprError::Parse(format!("bad number literal `{text}`")))?;
                toks.push(Tok::Num(value));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < bytes.len() {
                    let ch = bytes[i] as char;
                    if ch.is_ascii_alphanumeric() || ch == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Ident(src[start..i].to_string()));
            }
            other => {
                return Err(ExprError::Parse(format!("unexpected character `{other}`")));
            }
        }
    }
    Ok(toks)
}

/// Recursive-descent parser over the token stream.
struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// `expr := term (('+' | '-') term)*` — left-associative additive level.
    fn parse_expr(&mut self) -> Result<Ast, ExprError> {
        let mut node = self.parse_term()?;
        while let Some(tok) = self.peek() {
            match tok {
                Tok::Plus => {
                    self.next();
                    let rhs = self.parse_term()?;
                    node = Ast::Add(Box::new(node), Box::new(rhs));
                }
                Tok::Minus => {
                    self.next();
                    let rhs = self.parse_term()?;
                    node = Ast::Sub(Box::new(node), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    /// `term := unary (('*' | '/') unary)*` — left-associative multiplicative level.
    fn parse_term(&mut self) -> Result<Ast, ExprError> {
        let mut node = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            match tok {
                Tok::Star => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    node = Ast::Mul(Box::new(node), Box::new(rhs));
                }
                Tok::Slash => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    node = Ast::Div(Box::new(node), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    /// `unary := '-' unary | power` — unary minus binds looser than `**` (so
    /// `-a**2` parses as `-(a**2)`, matching Python/numexpr).
    fn parse_unary(&mut self) -> Result<Ast, ExprError> {
        if let Some(Tok::Minus) = self.peek() {
            self.next();
            let operand = self.parse_unary()?;
            return Ok(Ast::Neg(Box::new(operand)));
        }
        self.parse_power()
    }

    /// `power := atom ('**' unary)?` — right-associative power (the RHS recurses
    /// through `unary` so `a**-b` and `a**b**c` parse Python-style).
    fn parse_power(&mut self) -> Result<Ast, ExprError> {
        let base = self.parse_atom()?;
        if let Some(Tok::Pow) = self.peek() {
            self.next();
            let exp = self.parse_unary()?;
            return Ok(Ast::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    /// `atom := number | 'pi' | func '(' expr ')' | ident | '(' expr ')'`.
    fn parse_atom(&mut self) -> Result<Ast, ExprError> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Ast::Const(v)),
            Some(Tok::LParen) => {
                let inner = self.parse_expr()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(inner),
                    _ => Err(ExprError::Parse("expected `)`".into())),
                }
            }
            Some(Tok::Ident(name)) => {
                if let Some(func) = func_from_name(&name) {
                    match self.next() {
                        Some(Tok::LParen) => {}
                        _ => return Err(ExprError::Parse(format!("expected `(` after `{name}`"))),
                    }
                    let arg = self.parse_expr()?;
                    match self.next() {
                        Some(Tok::RParen) => Ok(Ast::Call(func, Box::new(arg))),
                        _ => Err(ExprError::Parse(format!("expected `)` closing `{name}(`"))),
                    }
                } else if name == "pi" {
                    Ok(Ast::Const(PI))
                } else {
                    Ok(Ast::Var(name))
                }
            }
            other => Err(ExprError::Parse(format!(
                "expected an atom, found {other:?}"
            ))),
        }
    }
}

/// Map a function name to its [`Func`], if it is one of the supported
/// transcendentals.
fn func_from_name(name: &str) -> Option<Func> {
    match name {
        "sin" => Some(Func::Sin),
        "cos" => Some(Func::Cos),
        "tan" => Some(Func::Tan),
        "sqrt" => Some(Func::Sqrt),
        "abs" => Some(Func::Abs),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn affine_formula() {
        let f = Formula::parse("scale * pos + offset").unwrap();
        assert_eq!(f.variables(), &["offset", "pos", "scale"]);
        let v = bind(&[("scale", 0.001), ("pos", 2.5), ("offset", 0.1)]);
        // Same f64 path numexpr took: 0.001*2.5 + 0.1.
        assert_eq!(f.eval(&v).unwrap(), 0.001 * 2.5 + 0.1);
    }

    #[test]
    fn precedence_add_after_mul() {
        let f = Formula::parse("a + b * c").unwrap();
        let v = bind(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        assert_eq!(f.eval(&v).unwrap(), 1.0 + 2.0 * 3.0);
    }

    #[test]
    fn unary_minus_binds_looser_than_pow() {
        // -a**2 == -(a**2), not (-a)**2.
        let f = Formula::parse("-a**2").unwrap();
        let v = bind(&[("a", 3.0)]);
        assert_eq!(f.eval(&v).unwrap(), -(3.0_f64.powi(2)));
    }

    #[test]
    fn integer_pow_uses_powi() {
        // a**6 distinguishes powi (exp-by-squaring) from a left-fold chain.
        let f = Formula::parse("a ** 6").unwrap();
        let a = 1.234567890123_f64;
        let v = bind(&[("a", a)]);
        assert_eq!(f.eval(&v).unwrap(), a.powi(6));
    }

    #[test]
    fn fractional_pow_uses_powf() {
        let f = Formula::parse("a ** 0.25").unwrap();
        let a = 1.234567890123_f64;
        let v = bind(&[("a", a)]);
        assert_eq!(f.eval(&v).unwrap(), a.powf(0.25));
    }

    #[test]
    fn pi_is_constant() {
        let f = Formula::parse("dist * pi").unwrap();
        assert_eq!(f.variables(), &["dist"]);
        let v = bind(&[("dist", 0.2)]);
        assert_eq!(f.eval(&v).unwrap(), 0.2 * PI);
    }

    #[test]
    fn transcendentals() {
        let f = Formula::parse("sin(a) + cos(a) - tan(a) + sqrt(b) + abs(c)").unwrap();
        let v = bind(&[("a", 0.37), ("b", 0.2), ("c", -0.5)]);
        let a = 0.37_f64;
        let expected = a.sin() + a.cos() - a.tan() + 0.2_f64.sqrt() + (-0.5_f64).abs();
        assert_eq!(f.eval(&v).unwrap(), expected);
    }

    #[test]
    fn unbound_variable_errors() {
        let f = Formula::parse("a + b").unwrap();
        let v = bind(&[("a", 1.0)]);
        assert_eq!(f.eval(&v), Err(ExprError::UnboundVariable("b".to_string())));
    }

    #[test]
    fn nested_parentheses() {
        let f = Formula::parse("(a + b) * (c - d)").unwrap();
        let v = bind(&[("a", 1.0), ("b", 2.0), ("c", 5.0), ("d", 1.0)]);
        assert_eq!(f.eval(&v).unwrap(), (1.0 + 2.0) * (5.0 - 1.0));
    }
}
