//! # String Functions Module
//!
//! This module provides string manipulation SQL functions:
//!
//! ## Character Functions
//! - `ASCII(str)` - Returns ASCII code of first character
//! - `CHAR_LENGTH(str)` / `CHARACTER_LENGTH(str)` - Character count
//! - `LENGTH(str)` - Byte length
//!
//! ## Case Conversion
//! - `UPPER(str)` / `UCASE(str)` - Convert to uppercase
//! - `LOWER(str)` / `LCASE(str)` - Convert to lowercase
//!
//! ## Substring Operations
//! - `LEFT(str, len)` - Left substring
//! - `RIGHT(str, len)` - Right substring
//! - `SUBSTR(str, pos, len)` / `SUBSTRING(str, pos, len)` / `MID(str, pos, len)`
//! - `SUBSTRING_INDEX(str, delim, count)` - Substring before/after delimiter
//!
//! ## Search Functions
//! - `INSTR(str, substr)` - Position of substring (1-based)
//! - `LOCATE(substr, str, pos)` / `POSITION(substr IN str)` - Find substring
//! - `FIELD(str, s1, s2, ...)` - Index of str in list
//! - `FIND_IN_SET(str, strlist)` - Position in comma-separated list
//!
//! ## Concatenation
//! - `CONCAT(s1, s2, ...)` - Concatenate strings
//! - `CONCAT_WS(sep, s1, s2, ...)` - Concatenate with separator
//!
//! ## Padding & Trimming
//! - `LPAD(str, len, pad)` - Left pad
//! - `RPAD(str, len, pad)` - Right pad
//! - `LTRIM(str)` - Trim left whitespace
//! - `RTRIM(str)` - Trim right whitespace
//! - `TRIM(str)` - Trim both sides
//!
//! ## Transformation
//! - `REPLACE(str, from, to)` - Replace occurrences
//! - `REVERSE(str)` - Reverse string
//! - `REPEAT(str, count)` - Repeat string
//! - `SPACE(n)` - Generate n spaces
//! - `INSERT(str, pos, len, newstr)` - Insert/replace substring
//!
//! ## Comparison
//! - `STRCMP(s1, s2)` - Compare strings (-1, 0, 1)

use crate::types::Value;
use std::borrow::Cow;

pub fn eval_string_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "ASCII" => eval_ascii(args),
        "CHAR_LENGTH" | "CHARACTER_LENGTH" => eval_char_length(args),
        "LENGTH" | "LEN" | "OCTET_LENGTH" => eval_length(args),
        "UPPER" | "UCASE" => eval_upper(args),
        "LOWER" | "LCASE" => eval_lower(args),
        "LEFT" => eval_left(args),
        "RIGHT" => eval_right(args),
        "SUBSTR" | "SUBSTRING" | "MID" => eval_substr(args),
        "SUBSTRING_INDEX" => eval_substring_index(args),
        "INSTR" => eval_instr(args),
        "LOCATE" | "POSITION" => eval_locate(args),
        "FIELD" => eval_field(args),
        "FIND_IN_SET" => eval_find_in_set(args),
        "CONCAT" => eval_concat(args),
        "CONCAT_WS" => eval_concat_ws(args),
        "LPAD" => eval_lpad(args),
        "RPAD" => eval_rpad(args),
        "LTRIM" => eval_ltrim(args),
        "RTRIM" => eval_rtrim(args),
        "TRIM" => eval_trim(args),
        "REPLACE" => eval_replace(args),
        "REVERSE" => eval_reverse(args),
        "REPEAT" => eval_repeat(args),
        "SPACE" => eval_space(args),
        "INSERT" => eval_insert(args),
        "STRCMP" => eval_strcmp(args),
        "FORMAT" => eval_format(args),
        _ => None,
    }
}

fn get_text<'a>(val: &Option<Value<'a>>) -> Option<Cow<'a, str>> {
    match val.as_ref()? {
        Value::Text(s) => Some(s.clone()),
        Value::Int(n) => Some(Cow::Owned(n.to_string())),
        Value::Float(f) => Some(Cow::Owned(f.to_string())),
        Value::Null => None,
        _ => None,
    }
}

fn get_int(val: &Option<Value>) -> Option<i64> {
    match val.as_ref()? {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        Value::Null => None,
        _ => None,
    }
}

fn eval_ascii<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let code = text.chars().next().map(|c| c as i64).unwrap_or(0);
    Some(Value::Int(code))
}

fn eval_char_length<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Int(text.chars().count() as i64))
}

fn eval_length<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Int(text.len() as i64))
}

fn eval_upper<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Text(Cow::Owned(text.to_uppercase())))
}

fn eval_lower<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Text(Cow::Owned(text.to_lowercase())))
}

fn eval_left<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let len = get_int(args.get(1)?)?;
    if len < 0 {
        return Some(Value::Text(Cow::Borrowed("")));
    }
    let result: String = text.chars().take(len as usize).collect();
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_right<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let len = get_int(args.get(1)?)?;
    if len < 0 {
        return Some(Value::Text(Cow::Borrowed("")));
    }
    let chars: Vec<char> = text.chars().collect();
    let skip = chars.len().saturating_sub(len as usize);
    let result: String = chars.into_iter().skip(skip).collect();
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_substr<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let pos = get_int(args.get(1)?)?;
    let len = args.get(2).and_then(get_int);
    
    let chars: Vec<char> = text.chars().collect();
    let start = if pos > 0 {
        (pos - 1) as usize
    } else if pos < 0 {
        chars.len().saturating_sub((-pos) as usize)
    } else {
        return Some(Value::Text(Cow::Borrowed("")));
    };
    
    let result: String = match len {
        Some(l) if l >= 0 => chars.iter().skip(start).take(l as usize).collect(),
        Some(_) => return Some(Value::Text(Cow::Borrowed(""))),
        None => chars.iter().skip(start).collect(),
    };
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_substring_index<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let delim = get_text(args.get(1)?)?;
    let count = get_int(args.get(2)?)?;
    
    if delim.is_empty() {
        return Some(Value::Text(Cow::Borrowed("")));
    }
    
    let parts: Vec<&str> = text.split(delim.as_ref()).collect();
    
    let result = if count > 0 {
        let take = (count as usize).min(parts.len());
        parts[..take].join(delim.as_ref())
    } else if count < 0 {
        let skip = parts.len().saturating_sub((-count) as usize);
        parts[skip..].join(delim.as_ref())
    } else {
        String::new()
    };
    
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_instr<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let haystack = get_text(args.first()?)?;
    let needle = get_text(args.get(1)?)?;
    
    let pos = haystack.find(needle.as_ref()).map(|p| p + 1).unwrap_or(0);
    Some(Value::Int(pos as i64))
}

fn eval_locate<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let needle = get_text(args.first()?)?;
    let haystack = get_text(args.get(1)?)?;
    let start = args.get(2).and_then(get_int).unwrap_or(1);
    
    if start < 1 {
        return Some(Value::Int(0));
    }
    
    let chars: Vec<char> = haystack.chars().collect();
    let search_start = (start - 1) as usize;
    
    if search_start >= chars.len() {
        return Some(Value::Int(0));
    }
    
    let search_str: String = chars[search_start..].iter().collect();
    let pos = search_str
        .find(needle.as_ref())
        .map(|p| {
            let byte_pos = p;
            let char_pos = search_str[..byte_pos].chars().count();
            char_pos + search_start + 1
        })
        .unwrap_or(0);
    
    Some(Value::Int(pos as i64))
}

fn eval_field<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.is_empty() {
        return Some(Value::Int(0));
    }
    
    let target = get_text(args.first()?)?;
    
    for (i, arg) in args.iter().skip(1).enumerate() {
        if let Some(val) = get_text(arg) {
            if val == target {
                return Some(Value::Int((i + 1) as i64));
            }
        }
    }
    
    Some(Value::Int(0))
}

fn eval_find_in_set<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let needle = get_text(args.first()?)?;
    let haystack = get_text(args.get(1)?)?;
    
    for (i, item) in haystack.split(',').enumerate() {
        if item == needle.as_ref() {
            return Some(Value::Int((i + 1) as i64));
        }
    }
    
    Some(Value::Int(0))
}

fn eval_concat<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let mut result = String::new();
    for arg in args {
        match arg.as_ref() {
            Some(Value::Text(s)) => result.push_str(s),
            Some(Value::Int(n)) => result.push_str(&n.to_string()),
            Some(Value::Float(f)) => result.push_str(&f.to_string()),
            Some(Value::Null) | None => return Some(Value::Null),
            _ => {}
        }
    }
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_concat_ws<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.is_empty() {
        return Some(Value::Null);
    }
    
    let sep = get_text(args.first()?)?;
    let mut parts: Vec<String> = Vec::new();
    
    for arg in args.iter().skip(1) {
        match arg.as_ref() {
            Some(Value::Text(s)) => parts.push(s.to_string()),
            Some(Value::Int(n)) => parts.push(n.to_string()),
            Some(Value::Float(f)) => parts.push(f.to_string()),
            Some(Value::Null) | None => continue,
            _ => {}
        }
    }
    
    Some(Value::Text(Cow::Owned(parts.join(sep.as_ref()))))
}

fn eval_lpad<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let target_len = get_int(args.get(1)?)? as usize;
    let pad = get_text(args.get(2)?)?;
    
    let char_count = text.chars().count();
    if char_count >= target_len {
        let result: String = text.chars().take(target_len).collect();
        return Some(Value::Text(Cow::Owned(result)));
    }
    
    if pad.is_empty() {
        return Some(Value::Text(Cow::Owned(text.to_string())));
    }
    
    let pad_chars: Vec<char> = pad.chars().collect();
    let pad_needed = target_len - char_count;
    let mut result = String::with_capacity(target_len);
    
    for i in 0..pad_needed {
        result.push(pad_chars[i % pad_chars.len()]);
    }
    result.push_str(&text);
    
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_rpad<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let target_len = get_int(args.get(1)?)? as usize;
    let pad = get_text(args.get(2)?)?;
    
    let char_count = text.chars().count();
    if char_count >= target_len {
        let result: String = text.chars().take(target_len).collect();
        return Some(Value::Text(Cow::Owned(result)));
    }
    
    if pad.is_empty() {
        return Some(Value::Text(Cow::Owned(text.to_string())));
    }
    
    let pad_chars: Vec<char> = pad.chars().collect();
    let pad_needed = target_len - char_count;
    let mut result = text.to_string();
    
    for i in 0..pad_needed {
        result.push(pad_chars[i % pad_chars.len()]);
    }
    
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_ltrim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Text(Cow::Owned(text.trim_start().to_string())))
}

fn eval_rtrim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Text(Cow::Owned(text.trim_end().to_string())))
}

fn eval_trim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    Some(Value::Text(Cow::Owned(text.trim().to_string())))
}

fn eval_replace<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let from = get_text(args.get(1)?)?;
    let to = get_text(args.get(2)?)?;
    
    Some(Value::Text(Cow::Owned(text.replace(from.as_ref(), to.as_ref()))))
}

fn eval_reverse<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let result: String = text.chars().rev().collect();
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_repeat<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let count = get_int(args.get(1)?)?;
    
    if count <= 0 {
        return Some(Value::Text(Cow::Borrowed("")));
    }
    
    Some(Value::Text(Cow::Owned(text.repeat(count as usize))))
}

fn eval_space<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let count = get_int(args.first()?)?;
    
    if count <= 0 {
        return Some(Value::Text(Cow::Borrowed("")));
    }
    
    Some(Value::Text(Cow::Owned(" ".repeat(count as usize))))
}

fn eval_insert<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let text = get_text(args.first()?)?;
    let pos = get_int(args.get(1)?)?;
    let len = get_int(args.get(2)?)?;
    let newstr = get_text(args.get(3)?)?;
    
    if pos < 1 || len < 0 {
        return Some(Value::Text(Cow::Owned(text.to_string())));
    }
    
    let chars: Vec<char> = text.chars().collect();
    let start = (pos - 1) as usize;
    
    if start > chars.len() {
        return Some(Value::Text(Cow::Owned(text.to_string())));
    }
    
    let end = (start + len as usize).min(chars.len());
    
    let mut result: String = chars[..start].iter().collect();
    result.push_str(&newstr);
    result.extend(chars[end..].iter());
    
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_strcmp<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let s1 = get_text(args.first()?)?;
    let s2 = get_text(args.get(1)?)?;
    
    let result = match s1.cmp(&s2) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    };
    
    Some(Value::Int(result))
}

fn eval_format<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let number = match args.first()?.as_ref()? {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let decimals = get_int(args.get(1)?)?.max(0) as usize;
    
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

    let result = if is_negative {
        format!("-{}", result)
    } else {
        result
    };
    
    Some(Value::Text(Cow::Owned(result)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii() {
        let args = vec![Some(Value::Text(Cow::Borrowed("A")))];
        assert_eq!(eval_ascii(&args), Some(Value::Int(65)));
    }

    #[test]
    fn test_char_length() {
        let args = vec![Some(Value::Text(Cow::Borrowed("hello")))];
        assert_eq!(eval_char_length(&args), Some(Value::Int(5)));
    }

    #[test]
    fn test_substring_index() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("www.mysql.com"))),
            Some(Value::Text(Cow::Borrowed("."))),
            Some(Value::Int(2)),
        ];
        assert_eq!(eval_substring_index(&args), Some(Value::Text(Cow::Owned("www.mysql".to_string()))));

        let args = vec![
            Some(Value::Text(Cow::Borrowed("www.mysql.com"))),
            Some(Value::Text(Cow::Borrowed("."))),
            Some(Value::Int(-2)),
        ];
        assert_eq!(eval_substring_index(&args), Some(Value::Text(Cow::Owned("mysql.com".to_string()))));
    }

    #[test]
    fn test_concat_ws() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed(","))),
            Some(Value::Text(Cow::Borrowed("a"))),
            Some(Value::Null),
            Some(Value::Text(Cow::Borrowed("b"))),
        ];
        assert_eq!(eval_concat_ws(&args), Some(Value::Text(Cow::Owned("a,b".to_string()))));
    }

    #[test]
    fn test_lpad() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("hi"))),
            Some(Value::Int(5)),
            Some(Value::Text(Cow::Borrowed("?!"))),
        ];
        assert_eq!(eval_lpad(&args), Some(Value::Text(Cow::Owned("?!?hi".to_string()))));
    }

    #[test]
    fn test_insert() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("Quadratic"))),
            Some(Value::Int(3)),
            Some(Value::Int(4)),
            Some(Value::Text(Cow::Borrowed("What"))),
        ];
        assert_eq!(eval_insert(&args), Some(Value::Text(Cow::Owned("QuWhattic".to_string()))));
    }

    #[test]
    fn test_field() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("b"))),
            Some(Value::Text(Cow::Borrowed("a"))),
            Some(Value::Text(Cow::Borrowed("b"))),
            Some(Value::Text(Cow::Borrowed("c"))),
        ];
        assert_eq!(eval_field(&args), Some(Value::Int(2)));
    }
}
