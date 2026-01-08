use crate::parsing::{parse_json_path, JsonNavigator};
use crate::records::jsonb::{JsonbValue, JsonbView};
use crate::sql::executor::ExecutorRow;
use crate::types::{OwnedValue, Value};
use std::borrow::Cow;

pub struct CompiledPredicate<'a> {
    expr: &'a crate::sql::ast::Expr<'a>,
    column_map: Vec<(String, usize)>,
    params: Option<Vec<OwnedValue>>,
    set_param_count: usize,
}

impl<'a> CompiledPredicate<'a> {
    pub fn new(expr: &'a crate::sql::ast::Expr<'a>, column_map: Vec<(String, usize)>) -> Self {
        Self {
            expr,
            column_map,
            params: None,
            set_param_count: 0,
        }
    }

    pub fn with_params(
        expr: &'a crate::sql::ast::Expr<'a>,
        column_map: Vec<(String, usize)>,
        params: &[OwnedValue],
        set_param_count: usize,
    ) -> Self {
        Self {
            expr,
            column_map,
            params: Some(params.to_vec()),
            set_param_count,
        }
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
            Expr::Like { .. } | Expr::Between { .. } | Expr::InList { .. } => {
                match self.eval_value(expr, row) {
                    Some(Value::Int(n)) => n != 0,
                    _ => false,
                }
            }
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
            Expr::Parameter(param_ref) => {
                if let Some(ref params) = self.params {
                    let idx = match param_ref {
                        crate::sql::ast::ParameterRef::Anonymous => self.set_param_count,
                        crate::sql::ast::ParameterRef::Positional(n) => {
                            if *n > 0 {
                                (*n - 1) as usize
                            } else {
                                return None;
                            }
                        }
                        crate::sql::ast::ParameterRef::Named(_) => self.set_param_count,
                    };
                    params.get(idx).map(|v| self.owned_value_to_value(v))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn owned_value_to_value(&self, v: &OwnedValue) -> Value<'a> {
        match v {
            OwnedValue::Null => Value::Null,
            OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
            OwnedValue::Int(i) => Value::Int(*i),
            OwnedValue::Float(f) => Value::Float(*f),
            OwnedValue::Text(s) => Value::Text(Cow::Owned(s.clone())),
            OwnedValue::Blob(b) => Value::Blob(Cow::Owned(b.clone())),
            OwnedValue::Vector(v) => Value::Vector(Cow::Owned(v.clone())),
            OwnedValue::Uuid(u) => Value::Uuid(*u),
            OwnedValue::MacAddr(m) => Value::MacAddr(*m),
            OwnedValue::Inet4(a) => Value::Inet4(*a),
            OwnedValue::Inet6(a) => Value::Inet6(*a),
            OwnedValue::Jsonb(b) => Value::Jsonb(Cow::Owned(b.clone())),
            OwnedValue::TimestampTz(micros, offset_secs) => Value::TimestampTz {
                micros: *micros,
                offset_secs: *offset_secs,
            },
            OwnedValue::Interval(micros, days, months) => Value::Interval {
                micros: *micros,
                days: *days,
                months: *months,
            },
            OwnedValue::Point(x, y) => Value::Point { x: *x, y: *y },
            OwnedValue::Box(low, high) => Value::GeoBox {
                low: *low,
                high: *high,
            },
            OwnedValue::Circle(center, radius) => Value::Circle {
                center: *center,
                radius: *radius,
            },
            OwnedValue::Enum(type_id, ordinal) => Value::Enum {
                type_id: *type_id,
                ordinal: *ordinal,
            },
            OwnedValue::Decimal(digits, scale) => Value::Decimal {
                digits: *digits,
                scale: *scale,
            },
            OwnedValue::ToastPointer(b) => Value::ToastPointer(Cow::Owned(b.clone())),
            OwnedValue::Date(_) | OwnedValue::Time(_) | OwnedValue::Timestamp(_) => Value::Null,
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
            DataType::Json | DataType::Jsonb => match val {
                Value::Text(s) => self.parse_and_build_jsonb(s),
                Value::Jsonb(_) => Some(val.clone()),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            DataType::Date => match val {
                Value::Text(s) => self.parse_date(s),
                Value::Int(d) => Some(Value::Int(*d)),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            DataType::Time => match val {
                Value::Text(s) => self.parse_time(s),
                Value::Int(t) => Some(Value::Int(*t)),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            DataType::Timestamp | DataType::TimestampTz => match val {
                Value::Text(s) => self.parse_timestamp(s),
                Value::TimestampTz {
                    micros,
                    offset_secs,
                } => Some(Value::TimestampTz {
                    micros: *micros,
                    offset_secs: *offset_secs,
                }),
                Value::Int(t) => Some(Value::TimestampTz {
                    micros: *t,
                    offset_secs: 0,
                }),
                Value::Null => Some(Value::Null),
                _ => None,
            },
            _ => Some(val.clone()),
        }
    }

    fn parse_date(&self, s: &str) -> Option<Value<'a>> {
        let s = s.trim();
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let year: i32 = parts[0].parse().ok()?;
        let month: u32 = parts[1].parse().ok()?;
        let day: u32 = parts[2].parse().ok()?;
        if !(1..=12).contains(&month) {
            return None;
        }
        let days_in_month = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                if is_leap {
                    29
                } else {
                    28
                }
            }
            _ => return None,
        };
        if !(1..=days_in_month).contains(&day) {
            return None;
        }
        let a = (14 - month as i32) / 12;
        let y = year + 4800 - a;
        let m = month as i32 + 12 * a - 3;
        let jdn = day as i32 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
        let days = jdn - 2440588;
        Some(Value::Int(days as i64))
    }

    fn parse_time(&self, s: &str) -> Option<Value<'a>> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }
        let hour: u32 = parts[0].parse().ok()?;
        let minute: u32 = parts[1].parse().ok()?;
        if hour > 23 || minute > 59 {
            return None;
        }
        let (second, micros_frac) = if parts.len() == 3 {
            let sec_parts: Vec<&str> = parts[2].split('.').collect();
            let sec: u32 = sec_parts[0].parse().ok()?;
            if sec > 59 {
                return None;
            }
            let frac: i64 = if sec_parts.len() > 1 {
                let frac_str = sec_parts[1];
                let padded = format!("{:0<6}", &frac_str[..frac_str.len().min(6)]);
                padded.parse().unwrap_or(0)
            } else {
                0
            };
            (sec, frac)
        } else {
            (0, 0)
        };
        let micros =
            (hour as i64 * 3600 + minute as i64 * 60 + second as i64) * 1_000_000 + micros_frac;
        Some(Value::Int(micros))
    }

    fn parse_timestamp(&self, s: &str) -> Option<Value<'a>> {
        let s = s.trim();
        let parts: Vec<&str> = s.splitn(2, [' ', 'T']).collect();
        if parts.is_empty() {
            return None;
        }
        let date_val = self.parse_date(parts[0])?;
        let days = match date_val {
            Value::Int(d) => d,
            _ => return None,
        };
        let time_micros = if parts.len() > 1 {
            match self.parse_time(parts[1]) {
                Some(Value::Int(t)) => t,
                _ => return None,
            }
        } else {
            0
        };
        let micros = days * 86400 * 1_000_000 + time_micros;
        Some(Value::TimestampTz {
            micros,
            offset_secs: 0,
        })
    }

    fn parse_and_build_jsonb(&self, s: &str) -> Option<Value<'a>> {
        use crate::records::jsonb::JsonbBuilder;

        let s = s.trim();
        if s == "null" {
            return Some(Value::Jsonb(Cow::Owned(JsonbBuilder::new_null().build())));
        }
        if s == "true" {
            return Some(Value::Jsonb(Cow::Owned(
                JsonbBuilder::new_bool(true).build(),
            )));
        }
        if s == "false" {
            return Some(Value::Jsonb(Cow::Owned(
                JsonbBuilder::new_bool(false).build(),
            )));
        }
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            let inner = &s[1..s.len() - 1];
            return Some(Value::Jsonb(Cow::Owned(
                JsonbBuilder::new_string(inner).build(),
            )));
        }
        if s.starts_with('{') || s.starts_with('[') {
            if let Some(jsonb_bytes) = self.parse_json_to_jsonb(s) {
                return Some(Value::Jsonb(Cow::Owned(jsonb_bytes)));
            }
            return None;
        }
        if let Ok(n) = s.parse::<f64>() {
            if s.chars().all(|c| {
                c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'
            }) {
                return Some(Value::Jsonb(Cow::Owned(
                    JsonbBuilder::new_number(n).build(),
                )));
            }
        }
        None
    }

    fn parse_json_to_jsonb(&self, s: &str) -> Option<Vec<u8>> {
        use crate::records::jsonb::{JsonbBuilder, JsonbBuilderValue};

        fn parse_value(s: &str) -> Option<JsonbBuilderValue> {
            let s = s.trim();
            if s == "null" {
                return Some(JsonbBuilderValue::Null);
            }
            if s == "true" {
                return Some(JsonbBuilderValue::Bool(true));
            }
            if s == "false" {
                return Some(JsonbBuilderValue::Bool(false));
            }
            if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                let inner = &s[1..s.len() - 1];
                return Some(JsonbBuilderValue::String(inner.to_string()));
            }
            if s.starts_with('{') && s.ends_with('}') {
                return parse_object(&s[1..s.len() - 1]);
            }
            if s.starts_with('[') && s.ends_with(']') {
                return parse_array(&s[1..s.len() - 1]);
            }
            if let Ok(n) = s.parse::<f64>() {
                if s.chars().all(|c| {
                    c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'
                }) {
                    return Some(JsonbBuilderValue::Number(n));
                }
            }
            None
        }

        fn parse_object(s: &str) -> Option<JsonbBuilderValue> {
            let s = s.trim();
            if s.is_empty() {
                return Some(JsonbBuilderValue::Object(Vec::new()));
            }
            let mut entries = Vec::new();
            let mut depth = 0;
            let mut in_string = false;
            let mut escape = false;
            let mut start = 0;

            let chars: Vec<char> = s.chars().collect();
            for (i, &c) in chars.iter().enumerate() {
                if escape {
                    escape = false;
                    continue;
                }
                if c == '\\' && in_string {
                    escape = true;
                    continue;
                }
                if c == '"' {
                    in_string = !in_string;
                } else if !in_string {
                    if c == '{' || c == '[' {
                        depth += 1;
                    } else if c == '}' || c == ']' {
                        depth -= 1;
                    } else if c == ',' && depth == 0 {
                        let part: String = chars[start..i].iter().collect();
                        if let Some((k, v)) = parse_key_value(&part) {
                            entries.push((k, v));
                        } else {
                            return None;
                        }
                        start = i + 1;
                    }
                }
            }
            if start < chars.len() {
                let part: String = chars[start..].iter().collect();
                if let Some((k, v)) = parse_key_value(&part) {
                    entries.push((k, v));
                } else {
                    return None;
                }
            }
            Some(JsonbBuilderValue::Object(entries))
        }

        fn parse_key_value(s: &str) -> Option<(String, JsonbBuilderValue)> {
            let s = s.trim();
            let colon_pos = s.find(':')?;
            let key_part = s[..colon_pos].trim();
            let val_part = s[colon_pos + 1..].trim();
            if key_part.starts_with('"') && key_part.ends_with('"') && key_part.len() >= 2 {
                let key = key_part[1..key_part.len() - 1].to_string();
                let value = parse_value(val_part)?;
                Some((key, value))
            } else {
                None
            }
        }

        fn parse_array(s: &str) -> Option<JsonbBuilderValue> {
            let s = s.trim();
            if s.is_empty() {
                return Some(JsonbBuilderValue::Array(Vec::new()));
            }
            let mut elements = Vec::new();
            let mut depth = 0;
            let mut in_string = false;
            let mut escape = false;
            let mut start = 0;

            let chars: Vec<char> = s.chars().collect();
            for (i, &c) in chars.iter().enumerate() {
                if escape {
                    escape = false;
                    continue;
                }
                if c == '\\' && in_string {
                    escape = true;
                    continue;
                }
                if c == '"' {
                    in_string = !in_string;
                } else if !in_string {
                    if c == '{' || c == '[' {
                        depth += 1;
                    } else if c == '}' || c == ']' {
                        depth -= 1;
                    } else if c == ',' && depth == 0 {
                        let part: String = chars[start..i].iter().collect();
                        if let Some(v) = parse_value(&part) {
                            elements.push(v);
                        } else {
                            return None;
                        }
                        start = i + 1;
                    }
                }
            }
            if start < chars.len() {
                let part: String = chars[start..].iter().collect();
                if let Some(v) = parse_value(&part) {
                    elements.push(v);
                } else {
                    return None;
                }
            }
            Some(JsonbBuilderValue::Array(elements))
        }

        fn build_jsonb(value: &JsonbBuilderValue) -> Vec<u8> {
            match value {
                JsonbBuilderValue::Null => JsonbBuilder::new_null().build(),
                JsonbBuilderValue::Bool(b) => JsonbBuilder::new_bool(*b).build(),
                JsonbBuilderValue::Number(n) => JsonbBuilder::new_number(*n).build(),
                JsonbBuilderValue::String(s) => JsonbBuilder::new_string(s.clone()).build(),
                JsonbBuilderValue::Array(elements) => {
                    let mut builder = JsonbBuilder::new_array();
                    for elem in elements {
                        builder.push(elem.clone());
                    }
                    builder.build()
                }
                JsonbBuilderValue::Object(entries) => {
                    let mut builder = JsonbBuilder::new_object();
                    for (key, val) in entries {
                        builder.set(key.clone(), val.clone());
                    }
                    builder.build()
                }
            }
        }

        let value = parse_value(s)?;
        Some(build_jsonb(&value))
    }

    fn eval_function(
        &self,
        func: &crate::sql::ast::FunctionCall<'a>,
        row: &ExecutorRow<'a>,
    ) -> Option<Value<'a>> {
        use crate::sql::ast::FunctionArgs;

        let func_name = func.name.name.to_uppercase();

        if func.over.is_none()
            && matches!(func_name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
        {
            let lookup_name = match &func.args {
                FunctionArgs::Star => func_name.to_lowercase(),
                FunctionArgs::Args(args) if args.len() == 1 => {
                    if let crate::sql::ast::Expr::Column(col) = args[0].value {
                        format!("{}_{}", func_name.to_lowercase(), col.column.to_lowercase())
                    } else {
                        func_name.to_lowercase()
                    }
                }
                _ => func_name.to_lowercase(),
            };

            let col_idx = self
                .column_map
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(&lookup_name))
                .or_else(|| {
                    self.column_map
                        .iter()
                        .find(|(name, _)| name.eq_ignore_ascii_case(&func_name.to_lowercase()))
                })
                .map(|(_, idx)| *idx)?;

            return row.get(col_idx).cloned();
        }

        let args: Vec<Option<Value<'a>>> = match &func.args {
            FunctionArgs::None | FunctionArgs::Star => vec![],
            FunctionArgs::Args(args) => args
                .iter()
                .map(|arg| self.eval_value(arg.value, row))
                .collect(),
        };

        crate::sql::functions::eval_function(&func_name, &args)
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
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => {
                let result = self.compare_values(&Some(left.clone()), &Some(right.clone()), op);
                Some(Value::Int(if result { 1 } else { 0 }))
            }
            BinaryOperator::And => {
                let l = self.value_to_bool(left);
                let r = self.value_to_bool(right);
                Some(Value::Int(if l && r { 1 } else { 0 }))
            }
            BinaryOperator::Or => {
                let l = self.value_to_bool(left);
                let r = self.value_to_bool(right);
                Some(Value::Int(if l || r { 1 } else { 0 }))
            }
        }
    }

    fn value_to_bool(&self, val: &Value<'a>) -> bool {
        match val {
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Null => false,
            Value::Text(s) => !s.is_empty(),
            _ => false,
        }
    }

    fn eval_json_extract(
        &self,
        json_val: &Value<'a>,
        key: &Value<'a>,
        as_text: bool,
    ) -> Option<Value<'a>> {
        match json_val {
            Value::Jsonb(bytes) => self.extract_from_jsonb_binary(bytes.as_ref(), key, as_text),
            Value::Text(s) => match key {
                Value::Text(key_str) => {
                    if key_str.starts_with('$') {
                        let path_elements = self.parse_json_path_elements(key_str)?;
                        self.traverse_json_path(s, &path_elements, as_text)
                    } else {
                        self.extract_json_key(s, key_str, as_text)
                    }
                }
                Value::Int(index) => self.extract_json_array_index(s, *index, as_text),
                _ => None,
            },
            _ => None,
        }
    }

    fn extract_from_jsonb_binary(
        &self,
        bytes: &[u8],
        key: &Value<'a>,
        as_text: bool,
    ) -> Option<Value<'a>> {
        let view = JsonbView::new(bytes).ok()?;

        let result = match key {
            Value::Text(key_str) => {
                if key_str.starts_with('$') {
                    let path_elements = self.parse_json_path_elements(key_str)?;
                    let path_refs: Vec<&str> = path_elements.iter().map(|s| s.as_str()).collect();
                    view.get_path(&path_refs).ok()?
                } else {
                    view.get(key_str).ok()?
                }
            }
            Value::Int(index) => view.array_get(*index as usize).ok()?,
            _ => return None,
        };

        match result {
            Some(jsonb_val) => self.jsonb_value_to_value(jsonb_val, as_text),
            None => Some(Value::Null),
        }
    }

    fn jsonb_value_to_value<'b>(&self, val: JsonbValue<'b>, as_text: bool) -> Option<Value<'a>> {
        if as_text {
            match val {
                JsonbValue::Null => Some(Value::Null),
                JsonbValue::Bool(b) => Some(Value::Text(Cow::Owned(b.to_string()))),
                JsonbValue::Number(n) => Some(Value::Text(Cow::Owned(n.to_string()))),
                JsonbValue::String(s) => Some(Value::Text(Cow::Owned(s.to_string()))),
                JsonbValue::Array(view) => match view.to_json_string() {
                    Ok(s) => Some(Value::Text(Cow::Owned(s))),
                    Err(_) => Some(Value::Null),
                },
                JsonbValue::Object(view) => match view.to_json_string() {
                    Ok(s) => Some(Value::Text(Cow::Owned(s))),
                    Err(_) => Some(Value::Null),
                },
            }
        } else {
            match val {
                JsonbValue::Null => Some(Value::Null),
                JsonbValue::Bool(b) => Some(Value::Int(if b { 1 } else { 0 })),
                JsonbValue::Number(n) => Some(Value::Float(n)),
                JsonbValue::String(s) => Some(Value::Text(Cow::Owned(s.to_string()))),
                JsonbValue::Array(view) => Some(Value::Jsonb(Cow::Owned(view.data().to_vec()))),
                JsonbValue::Object(view) => Some(Value::Jsonb(Cow::Owned(view.data().to_vec()))),
            }
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
        let path_str = match path {
            Value::Text(s) => s.as_ref(),
            _ => return None,
        };

        let path_elements = self.parse_json_path_elements(path_str)?;

        match json_val {
            Value::Jsonb(bytes) => {
                self.traverse_jsonb_path(bytes.as_ref(), &path_elements, as_text)
            }
            Value::Text(s) => self.traverse_json_path(s, &path_elements, as_text),
            _ => None,
        }
    }

    fn traverse_jsonb_path(
        &self,
        bytes: &[u8],
        path: &[String],
        as_text: bool,
    ) -> Option<Value<'a>> {
        let view = JsonbView::new(bytes).ok()?;

        if path.is_empty() {
            return self.jsonb_value_to_value(view.as_value().ok()?, as_text);
        }

        let path_refs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        let result = view.get_path(&path_refs).ok()?;

        match result {
            Some(jsonb_val) => self.jsonb_value_to_value(jsonb_val, as_text),
            None => Some(Value::Null),
        }
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
        let vec1 = self.value_to_vec(left)?;
        let vec2 = self.value_to_vec(right)?;

        if vec1.len() != vec2.len() {
            return None;
        }

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
        let vec1 = self.value_to_vec(left)?;
        let vec2 = self.value_to_vec(right)?;

        if vec1.len() != vec2.len() {
            return None;
        }

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
        let vec1 = self.value_to_vec(left)?;
        let vec2 = self.value_to_vec(right)?;

        if vec1.len() != vec2.len() {
            return None;
        }

        let dot: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();

        Some(Value::Float(dot as f64))
    }

    fn value_to_vec(&self, val: &Value<'a>) -> Option<Cow<'a, [f32]>> {
        match val {
            Value::Vector(v) => Some(v.clone()),
            Value::Text(s) => {
                let trimmed = s.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    let inner = &trimmed[1..trimmed.len() - 1];
                    let parsed: Result<Vec<f32>, _> = inner
                        .split(',')
                        .map(|x| x.trim().parse::<f32>())
                        .collect();
                    parsed.ok().map(Cow::Owned)
                } else {
                    None
                }
            }
            _ => None,
        }
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
