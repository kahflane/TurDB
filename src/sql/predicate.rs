use std::borrow::Cow;
use crate::sql::executor::ExecutorRow;
use crate::types::Value;

pub struct CompiledPredicate<'a> {
    expr: &'a crate::sql::ast::Expr<'a>,
    column_map: Vec<(String, usize)>,
}

impl<'a> CompiledPredicate<'a> {
    pub fn new(expr: &'a crate::sql::ast::Expr<'a>, column_map: Vec<(String, usize)>) -> Self {
        Self { expr, column_map }
    }

    pub fn evaluate(&self, row: &ExecutorRow<'a>) -> bool {
        self.eval_expr(self.expr, row)
    }

    fn eval_expr(&self, expr: &crate::sql::ast::Expr<'a>, row: &ExecutorRow<'a>) -> bool {
        use crate::sql::ast::{BinaryOperator, Expr, Literal};

        match expr {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => self.eval_expr(left, row) && self.eval_expr(right, row),
                BinaryOperator::Or => self.eval_expr(left, row) || self.eval_expr(right, row),
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq => {
                    let left_val = self.eval_value(left, row);
                    let right_val = self.eval_value(right, row);
                    self.compare_values(&left_val, &right_val, op)
                }
                _ => true,
            },
            Expr::Literal(Literal::Boolean(b)) => *b,
            _ => true,
        }
    }

    fn eval_value(
        &self,
        expr: &crate::sql::ast::Expr<'a>,
        row: &ExecutorRow<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::{Expr, Literal};

        match expr {
            Expr::Column(col_ref) => {
                let col_idx = self
                    .column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                    .map(|(_, idx)| *idx)?;
                row.get(col_idx).cloned()
            }
            Expr::Literal(lit) => Some(match lit {
                Literal::Integer(s) => Value::Int(s.parse().ok()?),
                Literal::Float(s) => Value::Float(s.parse().ok()?),
                Literal::String(s) => Value::Text(Cow::Borrowed(*s)),
                Literal::Boolean(b) => Value::Int(if *b { 1 } else { 0 }),
                Literal::Null => Value::Null,
                Literal::HexNumber(s) => {
                    Value::Int(i64::from_str_radix(s.trim_start_matches("0x"), 16).ok()?)
                }
                Literal::BinaryNumber(s) => {
                    Value::Int(i64::from_str_radix(s.trim_start_matches("0b"), 2).ok()?)
                }
            }),
            Expr::BinaryOp { left, op, right } => {
                let left_val = self.eval_value(left, row)?;
                let right_val = self.eval_value(right, row)?;
                self.eval_binary_op(&left_val, op, &right_val)
            }
            _ => None,
        }
    }

    fn eval_binary_op(
        &self,
        left: &Value<'a>,
        op: &crate::sql::ast::BinaryOperator,
        right: &Value<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::BinaryOperator;

        match op {
            BinaryOperator::Plus => self.eval_arithmetic_op(left, right, |a, b| a + b, |a, b| a + b),
            BinaryOperator::Minus => {
                self.eval_arithmetic_op(left, right, |a, b| a - b, |a, b| a - b)
            }
            BinaryOperator::Multiply => {
                self.eval_arithmetic_op(left, right, |a, b| a * b, |a, b| a * b)
            }
            BinaryOperator::Divide => {
                match (left, right) {
                    (Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a / b)),
                    (Value::Int(a), Value::Float(b)) if *b != 0.0 => {
                        Some(Value::Float(*a as f64 / b))
                    }
                    (Value::Float(a), Value::Int(b)) if *b != 0 => {
                        Some(Value::Float(a / *b as f64))
                    }
                    (Value::Float(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(a / b)),
                    _ => None,
                }
            }
            BinaryOperator::Modulo => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a % b)),
                (Value::Float(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(a % b)),
                (Value::Int(a), Value::Float(b)) if *b != 0.0 => {
                    Some(Value::Float(*a as f64 % b))
                }
                (Value::Float(a), Value::Int(b)) if *b != 0 => Some(Value::Float(a % *b as f64)),
                _ => None,
            },
            BinaryOperator::Power => match (left, right) {
                (Value::Int(a), Value::Int(b)) => {
                    if *b >= 0 {
                        Some(Value::Int(a.pow(*b as u32)))
                    } else {
                        Some(Value::Float((*a as f64).powi(*b as i32)))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Some(Value::Float(a.powf(*b))),
                (Value::Int(a), Value::Float(b)) => Some(Value::Float((*a as f64).powf(*b))),
                (Value::Float(a), Value::Int(b)) => Some(Value::Float(a.powi(*b as i32))),
                _ => None,
            },
            BinaryOperator::Concat => match (left, right) {
                (Value::Text(a), Value::Text(b)) => {
                    Some(Value::Text(Cow::Owned(format!("{}{}", a, b))))
                }
                _ => None,
            },
            BinaryOperator::BitwiseAnd => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Some(Value::Int(a & b)),
                _ => None,
            },
            BinaryOperator::BitwiseOr => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Some(Value::Int(a | b)),
                _ => None,
            },
            BinaryOperator::BitwiseXor => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Some(Value::Int(a ^ b)),
                _ => None,
            },
            BinaryOperator::LeftShift => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b >= 0 && *b < 64 => {
                    Some(Value::Int(a << (*b as u32)))
                }
                _ => None,
            },
            BinaryOperator::RightShift => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b >= 0 && *b < 64 => {
                    Some(Value::Int(a >> (*b as u32)))
                }
                _ => None,
            },
            BinaryOperator::JsonExtractText | BinaryOperator::JsonExtract => {
                self.eval_json_extract(left, right, *op == BinaryOperator::JsonExtractText)
            }
            BinaryOperator::JsonPathExtract | BinaryOperator::JsonPathExtractText => {
                self.eval_json_path_extract(left, right, *op == BinaryOperator::JsonPathExtractText)
            }
            BinaryOperator::VectorL2Distance => self.eval_vector_l2_distance(left, right),
            BinaryOperator::VectorCosineDistance => self.eval_vector_cosine_distance(left, right),
            BinaryOperator::VectorInnerProduct => self.eval_vector_inner_product(left, right),
            _ => None,
        }
    }

    fn eval_json_extract(
        &self,
        json_val: &Value<'a>,
        key: &Value<'a>,
        as_text: bool,
    ) -> Option<Value<'a>> {
        let json_bytes = match json_val {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };

        let json_str = std::str::from_utf8(json_bytes).ok()?;

        match key {
            Value::Text(key_str) => self.extract_json_key(json_str, key_str, as_text),
            Value::Int(index) => self.extract_json_array_index(json_str, *index, as_text),
            _ => None,
        }
    }

    fn extract_json_key(&self, json: &str, key: &str, as_text: bool) -> Option<Value<'a>> {
        let json = json.trim();
        if !json.starts_with('{') {
            return None;
        }

        let search_key = format!("\"{}\":", key);
        let key_pos = json.find(&search_key)?;
        let value_start = key_pos + search_key.len();
        let rest = json[value_start..].trim_start();

        let (value, _) = self.parse_json_value(rest)?;

        if as_text {
            match &value {
                Value::Text(s) => Some(Value::Text(s.clone())),
                Value::Int(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                Value::Float(f) => Some(Value::Text(Cow::Owned(f.to_string()))),
                Value::Null => Some(Value::Null),
                _ => None,
            }
        } else {
            Some(value)
        }
    }

    fn extract_json_array_index(&self, json: &str, index: i64, as_text: bool) -> Option<Value<'a>> {
        let json = json.trim();
        if !json.starts_with('[') {
            return None;
        }

        if index < 0 {
            return None;
        }

        let inner = &json[1..json.len().saturating_sub(1)];
        let mut current_idx = 0i64;
        let mut depth = 0;
        let mut start = 0;

        for (i, c) in inner.char_indices() {
            match c {
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    if current_idx == index {
                        let value_str = inner[start..i].trim();
                        let (value, _) = self.parse_json_value(value_str)?;
                        return if as_text {
                            match &value {
                                Value::Text(s) => Some(Value::Text(s.clone())),
                                Value::Int(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                                Value::Float(f) => Some(Value::Text(Cow::Owned(f.to_string()))),
                                Value::Null => Some(Value::Null),
                                _ => None,
                            }
                        } else {
                            Some(value)
                        };
                    }
                    current_idx += 1;
                    start = i + 1;
                }
                _ => {}
            }
        }

        if current_idx == index {
            let value_str = inner[start..].trim();
            let (value, _) = self.parse_json_value(value_str)?;
            return if as_text {
                match &value {
                    Value::Text(s) => Some(Value::Text(s.clone())),
                    Value::Int(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                    Value::Float(f) => Some(Value::Text(Cow::Owned(f.to_string()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            } else {
                Some(value)
            };
        }

        None
    }

    fn parse_json_value(&self, s: &str) -> Option<(Value<'a>, usize)> {
        let s = s.trim_start();
        if s.is_empty() {
            return None;
        }

        if let Some(rest) = s.strip_prefix('"') {
            let end = rest.find('"')?;
            let string_content = &rest[..end];
            Some((Value::Text(Cow::Owned(string_content.to_string())), end + 2))
        } else if s.starts_with("null") {
            Some((Value::Null, 4))
        } else if s.starts_with("true") {
            Some((Value::Int(1), 4))
        } else if s.starts_with("false") {
            Some((Value::Int(0), 5))
        } else if s.starts_with('{') || s.starts_with('[') {
            let open = s.chars().next()?;
            let close = if open == '{' { '}' } else { ']' };
            let mut depth = 1;
            let mut end = 1;
            for (i, c) in s[1..].char_indices() {
                if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 2;
                        break;
                    }
                }
            }
            let obj_str = &s[..end];
            Some((Value::Jsonb(Cow::Owned(obj_str.as_bytes().to_vec())), end))
        } else {
            let end = s
                .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                .unwrap_or(s.len());
            let num_str = &s[..end];
            if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                let f: f64 = num_str.parse().ok()?;
                Some((Value::Float(f), end))
            } else {
                let n: i64 = num_str.parse().ok()?;
                Some((Value::Int(n), end))
            }
        }
    }

    fn eval_json_path_extract(
        &self,
        json_val: &Value<'a>,
        _path: &Value<'a>,
        _as_text: bool,
    ) -> Option<Value<'a>> {
        let _json_bytes = match json_val {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };
        None
    }

    fn eval_vector_l2_distance(&self, left: &Value<'a>, right: &Value<'a>) -> Option<Value<'a>> {
        let (vec1, vec2) = match (left, right) {
            (Value::Vector(v1), Value::Vector(v2)) if v1.len() == v2.len() => {
                (v1.as_ref(), v2.as_ref())
            }
            _ => return None,
        };

        let sum: f32 = vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        Some(Value::Float(sum.sqrt() as f64))
    }

    fn eval_vector_cosine_distance(&self, left: &Value<'a>, right: &Value<'a>) -> Option<Value<'a>> {
        let (vec1, vec2) = match (left, right) {
            (Value::Vector(v1), Value::Vector(v2)) if v1.len() == v2.len() => {
                (v1.as_ref(), v2.as_ref())
            }
            _ => return None,
        };

        let dot: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return None;
        }

        let cosine_similarity = dot / (norm1 * norm2);
        let cosine_distance = 1.0 - cosine_similarity;

        Some(Value::Float(cosine_distance as f64))
    }

    fn eval_vector_inner_product(&self, left: &Value<'a>, right: &Value<'a>) -> Option<Value<'a>> {
        let (vec1, vec2) = match (left, right) {
            (Value::Vector(v1), Value::Vector(v2)) if v1.len() == v2.len() => {
                (v1.as_ref(), v2.as_ref())
            }
            _ => return None,
        };

        let dot: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        Some(Value::Float(-dot as f64))
    }

    fn eval_arithmetic_op<F, G>(
        &self,
        left: &Value<'a>,
        right: &Value<'a>,
        int_op: F,
        float_op: G,
    ) -> Option<Value<'a>>
    where
        F: Fn(i64, i64) -> i64,
        G: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Some(Value::Int(int_op(*a, *b))),
            (Value::Float(a), Value::Float(b)) => Some(Value::Float(float_op(*a, *b))),
            (Value::Int(a), Value::Float(b)) => Some(Value::Float(float_op(*a as f64, *b))),
            (Value::Float(a), Value::Int(b)) => Some(Value::Float(float_op(*a, *b as f64))),
            _ => None,
        }
    }

    fn compare_values(
        &self,
        left: &Option<Value<'a>>,
        right: &Option<Value<'a>>,
        op: &crate::sql::ast::BinaryOperator,
    ) -> bool {
        use crate::sql::ast::BinaryOperator;
        use std::cmp::Ordering;

        let (l, r) = match (left, right) {
            (Some(l), Some(r)) => (l, r),
            _ => return false,
        };

        let ordering = match (l, r) {
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) | (_, Value::Null) => None,
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            _ => None,
        };

        match (ordering, op) {
            (Some(Ordering::Equal), BinaryOperator::Eq) => true,
            (Some(Ordering::Equal), BinaryOperator::NotEq) => false,
            (Some(o), BinaryOperator::NotEq) if o != Ordering::Equal => true,
            (Some(Ordering::Less), BinaryOperator::Lt) => true,
            (Some(Ordering::Less), BinaryOperator::LtEq) => true,
            (Some(Ordering::Equal), BinaryOperator::LtEq) => true,
            (Some(Ordering::Greater), BinaryOperator::Gt) => true,
            (Some(Ordering::Greater), BinaryOperator::GtEq) => true,
            (Some(Ordering::Equal), BinaryOperator::GtEq) => true,
            _ => false,
        }
    }
}