use crate::parsing::{parse_json_path, JsonNavigator};
use crate::sql::executor::ExecutorRow;
use crate::types::Value;
use std::borrow::Cow;

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
                let lookup_name = if let Some(table) = col_ref.table {
                    format!("{}.{}", table, col_ref.column)
                } else {
                    col_ref.column.to_string()
                };
                let col_idx = self
                    .column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(&lookup_name))
                    .or_else(|| {
                        self.column_map
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                    })
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
            Expr::UnaryOp { op, expr } => {
                let val = self.eval_value(expr, row)?;
                self.eval_unary_op(op, &val)
            }
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => self.eval_case(operand.as_deref(), conditions, else_result.as_deref(), row),
            Expr::Cast { expr, data_type } => {
                let val = self.eval_value(expr, row)?;
                self.eval_cast(&val, data_type)
            }
            Expr::IsNull { expr, negated } => {
                let val = self.eval_value(expr, row);
                let is_null = matches!(val, Some(Value::Null) | None);
                let result = if *negated { !is_null } else { is_null };
                Some(Value::Int(if result { 1 } else { 0 }))
            }
            Expr::InList {
                expr,
                negated,
                list,
            } => {
                let target_val = self.eval_value(expr, row)?;
                let mut found = false;
                for list_item in list.iter() {
                    if let Some(list_val) = self.eval_value(list_item, row) {
                        if self.values_equal(&target_val, &list_val) {
                            found = true;
                            break;
                        }
                    }
                }
                let result = if *negated { !found } else { found };
                Some(Value::Int(if result { 1 } else { 0 }))
            }
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let val = self.eval_value(expr, row)?;
                let low_val = self.eval_value(low, row)?;
                let high_val = self.eval_value(high, row)?;
                let in_range = self
                    .value_cmp(&val, &low_val)
                    .is_some_and(|o| o != std::cmp::Ordering::Less)
                    && self
                        .value_cmp(&val, &high_val)
                        .is_some_and(|o| o != std::cmp::Ordering::Greater);
                let result = if *negated { !in_range } else { in_range };
                Some(Value::Int(if result { 1 } else { 0 }))
            }
            Expr::Like {
                expr,
                negated,
                pattern,
                escape: _,
                case_insensitive,
            } => {
                let val = self.eval_value(expr, row)?;
                let pat = self.eval_value(pattern, row)?;
                let matches = match (&val, &pat) {
                    (Value::Text(s), Value::Text(p)) => self.like_match(s, p, *case_insensitive),
                    _ => false,
                };
                let result = if *negated { !matches } else { matches };
                Some(Value::Int(if result { 1 } else { 0 }))
            }
            Expr::ArraySubscript { array, index } => {
                let array_val = self.eval_value(array, row)?;
                let index_val = self.eval_value(index, row)?;
                self.eval_array_subscript(&array_val, &index_val)
            }
            Expr::Function(func) => self.eval_function(func, row),
            _ => None,
        }
    }

    fn eval_unary_op(
        &self,
        op: &crate::sql::ast::UnaryOperator,
        val: &Value<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::UnaryOperator;

        match op {
            UnaryOperator::Minus => match val {
                Value::Int(n) => Some(Value::Int(-n)),
                Value::Float(f) => Some(Value::Float(-f)),
                _ => None,
            },
            UnaryOperator::Plus => match val {
                Value::Int(n) => Some(Value::Int(*n)),
                Value::Float(f) => Some(Value::Float(*f)),
                _ => None,
            },
            UnaryOperator::Not => match val {
                Value::Int(n) => Some(Value::Int(if *n == 0 { 1 } else { 0 })),
                _ => None,
            },
            UnaryOperator::BitwiseNot => match val {
                Value::Int(n) => Some(Value::Int(!n)),
                _ => None,
            },
        }
    }

    fn eval_case(
        &self,
        operand: Option<&crate::sql::ast::Expr<'a>>,
        conditions: &[crate::sql::ast::WhenClause<'a>],
        else_result: Option<&crate::sql::ast::Expr<'a>>,
        row: &ExecutorRow<'a>,
    ) -> Option<Value<'a>> {
        match operand {
            Some(op_expr) => {
                let op_val = self.eval_value(op_expr, row)?;
                for when_clause in conditions {
                    let when_val = self.eval_value(when_clause.condition, row)?;
                    if self.values_equal(&op_val, &when_val) {
                        return self.eval_value(when_clause.result, row);
                    }
                }
            }
            None => {
                for when_clause in conditions {
                    if self.eval_condition_as_bool(when_clause.condition, row) {
                        return self.eval_value(when_clause.result, row);
                    }
                }
            }
        }

        if let Some(else_expr) = else_result {
            self.eval_value(else_expr, row)
        } else {
            Some(Value::Null)
        }
    }

    fn eval_condition_as_bool(
        &self,
        expr: &crate::sql::ast::Expr<'a>,
        row: &ExecutorRow<'a>,
    ) -> bool {
        if let Some(val) = self.eval_value(expr, row) {
            match val {
                Value::Int(n) => n != 0,
                Value::Float(f) => f != 0.0,
                Value::Null => false,
                _ => false,
            }
        } else {
            false
        }
    }

    fn values_equal(&self, a: &Value<'a>, b: &Value<'a>) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
            (Value::Int(x), Value::Float(y)) => ((*x as f64) - y).abs() < f64::EPSILON,
            (Value::Float(x), Value::Int(y)) => (x - (*y as f64)).abs() < f64::EPSILON,
            (Value::Text(x), Value::Text(y)) => x == y,
            _ => false,
        }
    }

    fn value_cmp(&self, a: &Value<'a>, b: &Value<'a>) -> Option<std::cmp::Ordering> {
        match (a, b) {
            (Value::Null, _) | (_, Value::Null) => None,
            (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
            (Value::Int(x), Value::Float(y)) => (*x as f64).partial_cmp(y),
            (Value::Float(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)),
            (Value::Text(x), Value::Text(y)) => Some(x.cmp(y)),
            _ => None,
        }
    }

    fn like_match(&self, text: &str, pattern: &str, case_insensitive: bool) -> bool {
        let (text, pattern) = if case_insensitive {
            (text.to_lowercase(), pattern.to_lowercase())
        } else {
            (text.to_string(), pattern.to_string())
        };

        self.like_match_impl(text.as_bytes(), pattern.as_bytes())
    }

    fn like_match_impl(&self, text: &[u8], pattern: &[u8]) -> bool {
        let mut ti = 0;
        let mut pi = 0;
        let mut star_pi = None;
        let mut star_ti = 0;

        while ti < text.len() {
            if pi < pattern.len() && (pattern[pi] == b'_' || pattern[pi] == text[ti]) {
                ti += 1;
                pi += 1;
            } else if pi < pattern.len() && pattern[pi] == b'%' {
                star_pi = Some(pi);
                star_ti = ti;
                pi += 1;
            } else if let Some(sp) = star_pi {
                pi = sp + 1;
                star_ti += 1;
                ti = star_ti;
            } else {
                return false;
            }
        }

        while pi < pattern.len() && pattern[pi] == b'%' {
            pi += 1;
        }

        pi == pattern.len()
    }

    fn eval_array_subscript(&self, array: &Value<'a>, index: &Value<'a>) -> Option<Value<'a>> {
        let idx = match index {
            Value::Int(i) => *i,
            _ => return None,
        };

        let json_str = match array {
            Value::Text(s) if s.trim().starts_with('[') => s.as_ref(),
            Value::Jsonb(bytes) => {
                let s = std::str::from_utf8(bytes).ok()?;
                if s.trim().starts_with('[') {
                    s
                } else {
                    return Some(Value::Null);
                }
            }
            _ => return Some(Value::Null),
        };

        match self.extract_json_array_index(json_str, idx, false) {
            Some(val) => Some(val),
            None => Some(Value::Null),
        }
    }

    fn eval_cast(
        &self,
        val: &Value<'a>,
        data_type: &crate::sql::ast::DataType<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::DataType;

        match data_type {
            DataType::Integer | DataType::BigInt | DataType::SmallInt | DataType::TinyInt => {
                match val {
                    Value::Int(n) => Some(Value::Int(*n)),
                    Value::Float(f) => Some(Value::Int(*f as i64)),
                    Value::Text(s) => s.parse::<i64>().ok().map(Value::Int),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            DataType::Real
            | DataType::DoublePrecision
            | DataType::Decimal(_, _)
            | DataType::Numeric(_, _) => match val {
                Value::Int(n) => Some(Value::Float(*n as f64)),
                Value::Float(f) => Some(Value::Float(*f)),
                Value::Text(s) => s.parse::<f64>().ok().map(Value::Float),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            DataType::Text | DataType::Varchar(_) | DataType::Char(_) => match val {
                Value::Int(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                Value::Float(f) => Some(Value::Text(Cow::Owned(f.to_string()))),
                Value::Text(s) => Some(Value::Text(s.clone())),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            DataType::Boolean => match val {
                Value::Int(n) => Some(Value::Int(if *n != 0 { 1 } else { 0 })),
                Value::Float(f) => Some(Value::Int(if *f != 0.0 { 1 } else { 0 })),
                Value::Text(s) => {
                    let lower = s.to_lowercase();
                    if lower == "true"
                        || lower == "t"
                        || lower == "1"
                        || lower == "yes"
                        || lower == "on"
                    {
                        Some(Value::Int(1))
                    } else if lower == "false"
                        || lower == "f"
                        || lower == "0"
                        || lower == "no"
                        || lower == "off"
                    {
                        Some(Value::Int(0))
                    } else {
                        None
                    }
                }
                Value::Null => Some(Value::Null),
                _ => None,
            },
            _ => Some(val.clone()),
        }
    }

    fn eval_function(
        &self,
        func: &crate::sql::ast::FunctionCall<'a>,
        row: &ExecutorRow<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::FunctionArgs;

        let func_name = func.name.name.to_uppercase();
        let args: Vec<Option<Value<'a>>> = match &func.args {
            FunctionArgs::None | FunctionArgs::Star => vec![],
            FunctionArgs::Args(args) => args
                .iter()
                .map(|arg| self.eval_value(arg.value, row))
                .collect(),
        };

        match func_name.as_str() {
            "UPPER" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Text(Cow::Owned(s.to_uppercase()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "LOWER" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Text(Cow::Owned(s.to_lowercase()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "COALESCE" => {
                for arg in args {
                    match arg {
                        Some(Value::Null) | None => continue,
                        Some(val) => return Some(val),
                    }
                }
                Some(Value::Null)
            }
            "ABS" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Int(n) => Some(Value::Int(n.abs())),
                    Value::Float(f) => Some(Value::Float(f.abs())),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Int(s.chars().count() as i64)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "CONCAT" => {
                let mut result = String::new();
                for arg in args {
                    match arg {
                        Some(Value::Text(s)) => result.push_str(&s),
                        Some(Value::Int(n)) => result.push_str(&n.to_string()),
                        Some(Value::Float(f)) => result.push_str(&f.to_string()),
                        Some(Value::Null) | None => {}
                        _ => {}
                    }
                }
                Some(Value::Text(Cow::Owned(result)))
            }
            "TRIM" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim().to_string()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "LTRIM" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim_start().to_string()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "RTRIM" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim_end().to_string()))),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "SUBSTRING" | "SUBSTR" => {
                let text = match args.first()?.as_ref()? {
                    Value::Text(s) => s.as_ref(),
                    Value::Null => return Some(Value::Null),
                    _ => return None,
                };
                let start = match args.get(1)?.as_ref()? {
                    Value::Int(n) => (*n).max(1) as usize - 1,
                    _ => return None,
                };
                let len = if args.len() > 2 {
                    match args.get(2)?.as_ref()? {
                        Value::Int(n) => Some((*n).max(0) as usize),
                        _ => return None,
                    }
                } else {
                    None
                };
                let chars: Vec<char> = text.chars().collect();
                let result: String = if let Some(l) = len {
                    chars.iter().skip(start).take(l).collect()
                } else {
                    chars.iter().skip(start).collect()
                };
                Some(Value::Text(Cow::Owned(result)))
            }
            "NULLIF" => {
                if args.len() != 2 {
                    return None;
                }
                let first = args.first()?.as_ref()?;
                let second = args.get(1)?.as_ref()?;
                if self.values_equal(first, second) {
                    Some(Value::Null)
                } else {
                    Some(first.clone())
                }
            }
            "IFNULL" | "NVL" => {
                if args.len() != 2 {
                    return None;
                }
                let first = args.first()?;
                let second = args.get(1)?;
                match first {
                    Some(Value::Null) | None => second.clone(),
                    Some(val) => Some(val.clone()),
                }
            }
            "FLOOR" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Float(f) => Some(Value::Float(f.floor())),
                    Value::Int(n) => Some(Value::Int(*n)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "CEIL" | "CEILING" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Float(f) => Some(Value::Float(f.ceil())),
                    Value::Int(n) => Some(Value::Int(*n)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "ROUND" => {
                let arg = args.first()?.as_ref()?;
                let precision = if args.len() > 1 {
                    match args.get(1)?.as_ref()? {
                        Value::Int(n) => *n as i32,
                        _ => 0,
                    }
                } else {
                    0
                };
                match arg {
                    Value::Float(f) => {
                        let factor = 10f64.powi(precision);
                        Some(Value::Float((f * factor).round() / factor))
                    }
                    Value::Int(n) => Some(Value::Int(*n)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "SQRT" => {
                let arg = args.first()?.as_ref()?;
                match arg {
                    Value::Float(f) if *f >= 0.0 => Some(Value::Float(f.sqrt())),
                    Value::Int(n) if *n >= 0 => Some(Value::Float((*n as f64).sqrt())),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "POWER" | "POW" => {
                if args.len() != 2 {
                    return None;
                }
                let base = args.first()?.as_ref()?;
                let exp = args.get(1)?.as_ref()?;
                match (base, exp) {
                    (Value::Float(b), Value::Float(e)) => Some(Value::Float(b.powf(*e))),
                    (Value::Int(b), Value::Int(e)) if *e >= 0 => Some(Value::Int(b.pow(*e as u32))),
                    (Value::Int(b), Value::Float(e)) => Some(Value::Float((*b as f64).powf(*e))),
                    (Value::Float(b), Value::Int(e)) => Some(Value::Float(b.powi(*e as i32))),
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
            }
            "NOW" | "CURRENT_TIMESTAMP" => {
                use std::time::SystemTime;
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .ok()?;
                let secs = now.as_secs();
                let datetime = format_unix_timestamp_local(secs as i64);
                Some(Value::Text(Cow::Owned(datetime)))
            }
            "CURRENT_DATE" => {
                use std::time::SystemTime;
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .ok()?;
                let secs = now.as_secs();
                let date = format_unix_date_local(secs as i64);
                Some(Value::Text(Cow::Owned(date)))
            }
            "CURRENT_TIME" => {
                use std::time::SystemTime;
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .ok()?;
                let secs = now.as_secs();
                let time = format_unix_time_local(secs as i64);
                Some(Value::Text(Cow::Owned(time)))
            }
            "DATE_FORMAT" | "STRFTIME" => {
                if args.len() < 2 {
                    return None;
                }
                let format_str = match args.first()?.as_ref()? {
                    Value::Text(s) => s.to_string(),
                    _ => return None,
                };
                let datetime_str = match args.get(1)?.as_ref()? {
                    Value::Text(s) => s.to_string(),
                    Value::Null => return Some(Value::Null),
                    _ => return None,
                };
                let formatted = format_datetime_with_pattern(&datetime_str, &format_str);
                Some(Value::Text(Cow::Owned(formatted)))
            }
            "FORMAT" => {
                if args.len() < 2 {
                    return None;
                }
                let number = match args.first()?.as_ref()? {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    Value::Null => return Some(Value::Null),
                    _ => return None,
                };
                let decimals = match args.get(1)?.as_ref()? {
                    Value::Int(n) => *n as usize,
                    Value::Float(f) => *f as usize,
                    _ => return None,
                };
                let formatted = format_number_with_decimals(number, decimals);
                Some(Value::Text(Cow::Owned(formatted)))
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
            BinaryOperator::Plus => {
                self.eval_arithmetic_op(left, right, |a, b| a + b, |a, b| a + b)
            }
            BinaryOperator::Minus => {
                self.eval_arithmetic_op(left, right, |a, b| a - b, |a, b| a - b)
            }
            BinaryOperator::Multiply => {
                self.eval_arithmetic_op(left, right, |a, b| a * b, |a, b| a * b)
            }
            BinaryOperator::Divide => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a / b)),
                (Value::Int(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(*a as f64 / b)),
                (Value::Float(a), Value::Int(b)) if *b != 0 => Some(Value::Float(a / *b as f64)),
                (Value::Float(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(a / b)),
                _ => None,
            },
            BinaryOperator::Modulo => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a % b)),
                (Value::Float(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(a % b)),
                (Value::Int(a), Value::Float(b)) if *b != 0.0 => Some(Value::Float(*a as f64 % b)),
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
            BinaryOperator::JsonContains => Some(Value::Int(
                if self.eval_json_contains(left, right).unwrap_or(false) {
                    1
                } else {
                    0
                },
            )),
            BinaryOperator::JsonContainedBy => Some(Value::Int(
                if self.eval_json_contained_by(left, right).unwrap_or(false) {
                    1
                } else {
                    0
                },
            )),
            BinaryOperator::ArrayContains => Some(Value::Int(
                if self.eval_array_contains(left, right).unwrap_or(false) {
                    1
                } else {
                    0
                },
            )),
            BinaryOperator::ArrayContainedBy => Some(Value::Int(
                if self.eval_array_contained_by(left, right).unwrap_or(false) {
                    1
                } else {
                    0
                },
            )),
            BinaryOperator::ArrayOverlaps => Some(Value::Int(
                if self.eval_array_overlaps(left, right).unwrap_or(false) {
                    1
                } else {
                    0
                },
            )),
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

    pub fn eval_json_path_extract(
        &self,
        json_val: &Value<'a>,
        path: &Value<'a>,
        as_text: bool,
    ) -> Option<Value<'a>> {
        let json_bytes = match json_val {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };

        let path_str = match path {
            Value::Text(s) => s.as_ref(),
            _ => return None,
        };

        let path_elements = self.parse_json_path_elements(path_str)?;
        let json_str = std::str::from_utf8(json_bytes).ok()?;

        self.traverse_json_path(json_str, &path_elements, as_text)
    }

    fn parse_json_path_elements(&self, path: &str) -> Option<Vec<String>> {
        parse_json_path(path)
    }

    fn traverse_json_path(&self, json: &str, path: &[String], as_text: bool) -> Option<Value<'a>> {
        let mut current = json.trim().to_string();

        for key in path {
            current = self.extract_at_key_or_index(&current, key)?;
        }

        let (value, _) = self.parse_json_value(&current)?;

        if as_text {
            match &value {
                Value::Text(s) => Some(Value::Text(Cow::Owned(s.to_string()))),
                Value::Int(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                Value::Float(f) => Some(Value::Text(Cow::Owned(f.to_string()))),
                Value::Null => Some(Value::Null),
                Value::Jsonb(bytes) => Some(Value::Text(Cow::Owned(
                    String::from_utf8_lossy(bytes).to_string(),
                ))),
                _ => None,
            }
        } else {
            Some(value)
        }
    }

    fn extract_at_key_or_index(&self, json: &str, key: &str) -> Option<String> {
        let json = json.trim();

        if let Ok(index) = key.parse::<usize>() {
            self.extract_array_element(json, index)
        } else {
            self.extract_object_key(json, key)
        }
    }

    fn extract_object_key(&self, json: &str, key: &str) -> Option<String> {
        let json = json.trim();
        if !json.starts_with('{') {
            return None;
        }

        let search_key = format!("\"{}\":", key);
        let key_pos = json.find(&search_key)?;
        let value_start = key_pos + search_key.len();
        let rest = json[value_start..].trim_start();

        let (_, len) = self.parse_json_value(rest)?;
        Some(rest[..len].to_string())
    }

    fn extract_array_element(&self, json: &str, index: usize) -> Option<String> {
        let json = json.trim();
        if !json.starts_with('[') {
            return None;
        }

        let inner = &json[1..json.len().saturating_sub(1)];
        let mut current_idx = 0;
        let mut depth = 0;
        let mut start = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in inner.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' | '{' if !in_string => depth += 1,
                ']' | '}' if !in_string => depth -= 1,
                ',' if depth == 0 && !in_string => {
                    if current_idx == index {
                        return Some(inner[start..i].trim().to_string());
                    }
                    current_idx += 1;
                    start = i + 1;
                }
                _ => {}
            }
        }

        if current_idx == index {
            return Some(inner[start..].trim().to_string());
        }

        None
    }

    pub fn eval_json_contains(&self, left: &Value<'a>, right: &Value<'a>) -> Option<bool> {
        let left_bytes = match left {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };
        let right_bytes = match right {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };

        let left_str = std::str::from_utf8(left_bytes).ok()?;
        let right_str = std::str::from_utf8(right_bytes).ok()?;

        Some(self.json_contains_impl(left_str.trim(), right_str.trim()))
    }

    fn eval_json_contained_by(&self, left: &Value<'a>, right: &Value<'a>) -> Option<bool> {
        self.eval_json_contains(right, left)
    }

    fn json_contains_impl(&self, container: &str, contained: &str) -> bool {
        let nav = JsonNavigator::new("");
        nav.json_contains(container, contained)
    }

    fn json_values_equal(&self, a: &str, b: &str) -> bool {
        let a = a.trim();
        let b = b.trim();

        if a == b {
            return true;
        }

        let a_parsed = self.parse_json_value(a);
        let b_parsed = self.parse_json_value(b);

        match (a_parsed, b_parsed) {
            (Some((Value::Int(x), _)), Some((Value::Int(y), _))) => x == y,
            (Some((Value::Float(x), _)), Some((Value::Float(y), _))) => {
                (x - y).abs() < f64::EPSILON
            }
            (Some((Value::Int(x), _)), Some((Value::Float(y), _))) => {
                ((x as f64) - y).abs() < f64::EPSILON
            }
            (Some((Value::Float(x), _)), Some((Value::Int(y), _))) => {
                (x - (y as f64)).abs() < f64::EPSILON
            }
            (Some((Value::Text(x), _)), Some((Value::Text(y), _))) => x == y,
            (Some((Value::Null, _)), Some((Value::Null, _))) => true,
            _ => false,
        }
    }

    fn parse_array_elements_local(&self, json: &str) -> Vec<String> {
        let nav = JsonNavigator::new("");
        nav.parse_array_elements(json)
    }

    pub fn eval_array_contains(&self, left: &Value<'a>, right: &Value<'a>) -> Option<bool> {
        let left_bytes = match left {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };
        let right_bytes = match right {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };

        let left_str = std::str::from_utf8(left_bytes).ok()?;
        let right_str = std::str::from_utf8(right_bytes).ok()?;

        let left_elements = self.parse_array_elements_local(left_str.trim());
        let right_elements = self.parse_array_elements_local(right_str.trim());

        if left_elements.is_empty() && !left_str.trim().starts_with('[') {
            return None;
        }
        if right_elements.is_empty() && !right_str.trim().starts_with('[') {
            return None;
        }

        for elem in &right_elements {
            let found = left_elements
                .iter()
                .any(|l| self.json_values_equal(l, elem));
            if !found {
                return Some(false);
            }
        }
        Some(true)
    }

    fn eval_array_contained_by(&self, left: &Value<'a>, right: &Value<'a>) -> Option<bool> {
        self.eval_array_contains(right, left)
    }

    pub fn eval_array_overlaps(&self, left: &Value<'a>, right: &Value<'a>) -> Option<bool> {
        let left_bytes = match left {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };
        let right_bytes = match right {
            Value::Jsonb(bytes) => bytes.as_ref(),
            Value::Text(s) => s.as_bytes(),
            _ => return None,
        };

        let left_str = std::str::from_utf8(left_bytes).ok()?;
        let right_str = std::str::from_utf8(right_bytes).ok()?;

        let left_elements = self.parse_array_elements_local(left_str.trim());
        let right_elements = self.parse_array_elements_local(right_str.trim());

        if left_elements.is_empty() && !left_str.trim().starts_with('[') {
            return None;
        }
        if right_elements.is_empty() && !right_str.trim().starts_with('[') {
            return None;
        }

        for left_elem in &left_elements {
            for right_elem in &right_elements {
                if self.json_values_equal(left_elem, right_elem) {
                    return Some(true);
                }
            }
        }
        Some(false)
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

    fn eval_vector_cosine_distance(
        &self,
        left: &Value<'a>,
        right: &Value<'a>,
    ) -> Option<Value<'a>> {
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

    pub fn evaluate_to_value(&self, row: &ExecutorRow<'a>) -> Option<Value<'a>> {
        self.eval_value(self.expr, row)
    }
}

pub struct CompiledProjection<'a> {
    expressions: Vec<&'a crate::sql::ast::Expr<'a>>,
    column_map: Vec<(String, usize)>,
}

impl<'a> CompiledProjection<'a> {
    pub fn new(
        expressions: Vec<&'a crate::sql::ast::Expr<'a>>,
        column_map: Vec<(String, usize)>,
    ) -> Self {
        Self {
            expressions,
            column_map,
        }
    }

    pub fn evaluate(&self, row: &ExecutorRow<'a>) -> Vec<Option<Value<'a>>> {
        self.expressions
            .iter()
            .map(|expr| {
                let pred = CompiledPredicate::new(expr, self.column_map.clone());
                pred.evaluate_to_value(row)
            })
            .collect()
    }

    pub fn expression_count(&self) -> usize {
        self.expressions.len()
    }
}

fn get_local_timezone_offset() -> i64 {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    #[cfg(unix)]
    {
        use std::mem::MaybeUninit;
        
        let mut tm = MaybeUninit::<libc::tm>::uninit();
        let time_t = now as libc::time_t;
        
        unsafe {
            if libc::localtime_r(&time_t, tm.as_mut_ptr()).is_null() {
                return 0;
            }
            let tm = tm.assume_init();
            tm.tm_gmtoff as i64
        }
    }
    
    #[cfg(not(unix))]
    {
        0
    }
}

fn format_unix_timestamp_local(secs: i64) -> String {
    let offset = get_local_timezone_offset();
    format_unix_timestamp(secs + offset)
}

fn format_unix_timestamp(secs: i64) -> String {
    const SECONDS_PER_DAY: i64 = 86400;
    const DAYS_PER_YEAR: i64 = 365;
    const DAYS_PER_4_YEARS: i64 = 1461;
    const DAYS_PER_100_YEARS: i64 = 36524;
    const DAYS_PER_400_YEARS: i64 = 146097;

    let mut days = secs / SECONDS_PER_DAY;
    let day_secs = secs % SECONDS_PER_DAY;

    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;

    days += 719468;

    let era = if days >= 0 {
        days / DAYS_PER_400_YEARS
    } else {
        (days - DAYS_PER_400_YEARS + 1) / DAYS_PER_400_YEARS
    };
    let doe = (days - era * DAYS_PER_400_YEARS) as u32;
    let yoe = (doe - doe / DAYS_PER_4_YEARS as u32 + doe / DAYS_PER_100_YEARS as u32
        - doe / DAYS_PER_400_YEARS as u32)
        / DAYS_PER_YEAR as u32;
    let y = yoe as i64 + era * 400;
    let doy = doe
        - (DAYS_PER_YEAR as u32 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        y, m, d, hours, minutes, seconds
    )
}

fn format_unix_date_local(secs: i64) -> String {
    let ts = format_unix_timestamp_local(secs);
    ts.split(' ').next().unwrap_or("").to_string()
}

fn format_unix_time_local(secs: i64) -> String {
    let ts = format_unix_timestamp_local(secs);
    ts.split(' ').nth(1).unwrap_or("").to_string()
}

fn format_datetime_with_pattern(datetime_str: &str, pattern: &str) -> String {
    let parts: Vec<&str> = datetime_str.split(' ').collect();
    let date_part = parts.first().unwrap_or(&"");
    let time_part = parts.get(1).unwrap_or(&"00:00:00");

    let date_parts: Vec<&str> = date_part.split('-').collect();
    let year = date_parts.first().unwrap_or(&"0000");
    let month = date_parts.get(1).unwrap_or(&"01");
    let day = date_parts.get(2).unwrap_or(&"01");

    let time_parts: Vec<&str> = time_part.split(':').collect();
    let hour = time_parts.first().unwrap_or(&"00");
    let minute = time_parts.get(1).unwrap_or(&"00");
    let second = time_parts.get(2).unwrap_or(&"00");

    let hour_num: u32 = hour.parse().unwrap_or(0);
    let hour_12 = if hour_num == 0 {
        12
    } else if hour_num > 12 {
        hour_num - 12
    } else {
        hour_num
    };
    let am_pm = if hour_num < 12 { "AM" } else { "PM" };

    let month_num: u32 = month.parse().unwrap_or(1);
    let month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ];
    let month_abbrs = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_name = month_names.get((month_num as usize).saturating_sub(1)).unwrap_or(&"");
    let month_abbr = month_abbrs.get((month_num as usize).saturating_sub(1)).unwrap_or(&"");

    pattern
        .replace("%Y", year)
        .replace("%y", &year[year.len().saturating_sub(2)..])
        .replace("%m", month)
        .replace("%d", day)
        .replace("%H", hour)
        .replace("%i", minute)
        .replace("%s", second)
        .replace("%M", month_name)
        .replace("%b", month_abbr)
        .replace("%h", &format!("{:02}", hour_12))
        .replace("%p", am_pm)
        .replace("%T", &format!("{}:{}:{}", hour, minute, second))
        .replace("%D", &format!("{}{}", day, ordinal_suffix(day.parse().unwrap_or(1))))
}

fn ordinal_suffix(n: u32) -> &'static str {
    match n % 100 {
        11 | 12 | 13 => "th",
        _ => match n % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    }
}

fn format_number_with_decimals(number: f64, decimals: usize) -> String {
    let formatted = format!("{:.prec$}", number, prec = decimals);
    
    let parts: Vec<&str> = formatted.split('.').collect();
    let integer_part = parts.first().unwrap_or(&"0");
    let decimal_part = parts.get(1);

    let is_negative = integer_part.starts_with('-');
    let abs_integer: String = integer_part.chars().filter(|c| c.is_ascii_digit()).collect();
    
    let mut with_commas = String::new();
    for (i, c) in abs_integer.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            with_commas.push(',');
        }
        with_commas.push(c);
    }
    let with_commas: String = with_commas.chars().rev().collect();

    let result = match decimal_part {
        Some(dec) => format!("{}.{}", with_commas, dec),
        None => with_commas,
    };

    if is_negative {
        format!("-{}", result)
    } else {
        result
    }
}
