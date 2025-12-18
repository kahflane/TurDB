//! # ASCII Table Formatter
//!
//! Renders query results as ASCII tables with box-drawing characters.
//! This provides a familiar MySQL-style output format.
//!
//! ## Output Format
//!
//! ```text
//! +----+-------+-----+
//! | id | name  | age |
//! +----+-------+-----+
//! |  1 | Alice |  30 |
//! |  2 | Bob   |  25 |
//! +----+-------+-----+
//! 2 rows in set (0.001 sec)
//! ```
//!
//! ## Column Width Calculation
//!
//! Column widths are calculated as the maximum of:
//! - Header length
//! - Maximum value length in that column
//! - Minimum width of 1 character
//!
//! ## Value Formatting
//!
//! Values are formatted according to their type:
//! - NULL: displayed as "NULL"
//! - Text: displayed as-is
//! - Numbers: right-aligned
//! - Blobs: displayed as hex with length limit
//! - Vectors: displayed as "[f32; N]" summary
//!
//! ## Performance
//!
//! The formatter makes two passes:
//! 1. Calculate column widths (requires iterating all values)
//! 2. Render output
//!
//! For large result sets, consider using LIMIT to reduce output.
//!
//! ## Implementation Notes
//!
//! - Uses `+`, `-`, and `|` for box drawing (ASCII compatible)
//! - Pads values with spaces for alignment
//! - Numbers are right-aligned, text is left-aligned
//! - Maximum column width is 50 characters (truncated with "...")

use crate::database::Row;
use crate::types::OwnedValue;
use std::fmt::Write;

const MAX_COLUMN_WIDTH: usize = 50;
const BLOB_PREVIEW_BYTES: usize = 16;

pub struct TableFormatter {
    headers: Vec<String>,
    widths: Vec<usize>,
    rows: Vec<Vec<String>>,
}

impl TableFormatter {
    pub fn new(headers: Vec<String>, rows: &[Row]) -> Self {
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len().max(1)).collect();

        let formatted_rows: Vec<Vec<String>> = rows
            .iter()
            .map(|row| {
                row.values
                    .iter()
                    .enumerate()
                    .map(|(i, val)| {
                        let formatted = format_value(val);
                        if i < widths.len() {
                            widths[i] = widths[i].max(formatted.len()).min(MAX_COLUMN_WIDTH);
                        }
                        formatted
                    })
                    .collect()
            })
            .collect();

        Self {
            headers,
            widths,
            rows: formatted_rows,
        }
    }

    pub fn render(&self) -> String {
        let mut output = String::new();

        self.write_separator(&mut output);
        self.write_header_row(&mut output);
        self.write_separator(&mut output);

        for row in &self.rows {
            self.write_data_row(&mut output, row);
        }

        self.write_separator(&mut output);

        output
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn write_separator(&self, output: &mut String) {
        output.push('+');
        for width in &self.widths {
            for _ in 0..(*width + 2) {
                output.push('-');
            }
            output.push('+');
        }
        output.push('\n');
    }

    fn write_header_row(&self, output: &mut String) {
        output.push('|');
        for (i, header) in self.headers.iter().enumerate() {
            let width = self.widths.get(i).copied().unwrap_or(1);
            let _ = write!(output, " {:<width$} |", truncate(header, width), width = width);
        }
        output.push('\n');
    }

    fn write_data_row(&self, output: &mut String, row: &[String]) {
        output.push('|');
        for (i, value) in row.iter().enumerate() {
            let width = self.widths.get(i).copied().unwrap_or(1);
            let truncated = truncate(value, width);
            let _ = write!(output, " {:<width$} |", truncated, width = width);
        }
        output.push('\n');
    }
}

fn format_value(value: &OwnedValue) -> String {
    match value {
        OwnedValue::Null => "NULL".to_string(),
        OwnedValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        OwnedValue::Int(i) => i.to_string(),
        OwnedValue::Float(f) => format!("{:.6}", f).trim_end_matches('0').trim_end_matches('.').to_string(),
        OwnedValue::Text(s) => s.clone(),
        OwnedValue::Blob(b) => format_blob(b),
        OwnedValue::Uuid(u) => format_uuid(u),
        OwnedValue::Jsonb(j) => format_jsonb(j),
        OwnedValue::Vector(v) => format_vector(v),
        OwnedValue::Date(d) => format_date(*d),
        OwnedValue::Time(t) => format_time(*t),
        OwnedValue::Timestamp(ts) => format_timestamp(*ts),
        OwnedValue::TimestampTz(ts, _offset) => format_timestamp(*ts),
        OwnedValue::Interval(micros, days, months) => format_interval(*micros, *days, *months),
        OwnedValue::MacAddr(m) => format_macaddr(m),
        OwnedValue::Inet4(ip) => format_inet4(ip),
        OwnedValue::Inet6(ip) => format_inet6(ip),
        OwnedValue::Point(x, y) => format!("({}, {})", x, y),
        OwnedValue::Box(p1, p2) => format!("(({}, {}), ({}, {}))", p1.0, p1.1, p2.0, p2.1),
        OwnedValue::Circle(center, radius) => format!("<({}, {}), {}>", center.0, center.1, radius),
        OwnedValue::Decimal(value, scale) => format_decimal(*value, *scale),
        OwnedValue::Enum(type_id, variant) => format!("enum({}, {})", type_id, variant),
    }
}

fn format_blob(bytes: &[u8]) -> String {
    if bytes.len() <= BLOB_PREVIEW_BYTES {
        let hex: String = bytes.iter().map(|b| format!("{:02X}", b)).collect();
        format!("x'{}'", hex)
    } else {
        let hex: String = bytes[..BLOB_PREVIEW_BYTES].iter().map(|b| format!("{:02X}", b)).collect();
        format!("x'{}'... ({} bytes)", hex, bytes.len())
    }
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

fn format_jsonb(bytes: &[u8]) -> String {
    format!("<jsonb: {} bytes>", bytes.len())
}

fn format_vector(floats: &[f32]) -> String {
    if floats.len() <= 4 {
        let parts: Vec<String> = floats.iter().map(|f| format!("{:.3}", f)).collect();
        format!("[{}]", parts.join(", "))
    } else {
        format!("[f32; {}]", floats.len())
    }
}

fn format_date(days_since_epoch: i32) -> String {
    let epoch = 719528;
    let total_days = epoch + days_since_epoch;

    let (year, month, day) = days_to_ymd(total_days);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn format_time(micros: i64) -> String {
    let total_seconds = micros / 1_000_000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let micros_part = micros % 1_000_000;

    if micros_part == 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}:{:02}.{:06}", hours, minutes, seconds, micros_part)
    }
}

fn format_timestamp(micros: i64) -> String {
    let seconds = micros / 1_000_000;
    let micros_part = (micros % 1_000_000).abs();

    let days = (seconds / 86400) as i32;
    let time_of_day = (seconds % 86400).abs();

    let epoch = 719528;
    let total_days = epoch + days;
    let (year, month, day) = days_to_ymd(total_days);

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let secs = time_of_day % 60;

    if micros_part == 0 {
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", year, month, day, hours, minutes, secs)
    } else {
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:06}", year, month, day, hours, minutes, secs, micros_part)
    }
}

fn format_interval(micros: i64, days: i32, months: i32) -> String {
    let mut parts = Vec::new();

    if months != 0 {
        let years = months / 12;
        let remaining_months = months % 12;
        if years != 0 {
            parts.push(format!("{} year{}", years, if years.abs() == 1 { "" } else { "s" }));
        }
        if remaining_months != 0 {
            parts.push(format!("{} mon{}", remaining_months, if remaining_months.abs() == 1 { "" } else { "s" }));
        }
    }

    if days != 0 {
        parts.push(format!("{} day{}", days, if days.abs() == 1 { "" } else { "s" }));
    }

    if micros != 0 {
        let total_seconds = micros / 1_000_000;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        parts.push(format!("{:02}:{:02}:{:02}", hours, minutes, seconds));
    }

    if parts.is_empty() {
        "00:00:00".to_string()
    } else {
        parts.join(" ")
    }
}

fn days_to_ymd(total_days: i32) -> (i32, u32, u32) {
    let a = total_days + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - (146097 * b) / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - (1461 * d) / 4;
    let m = (5 * e + 2) / 153;

    let day = (e - (153 * m + 2) / 5 + 1) as u32;
    let month = (m + 3 - 12 * (m / 10)) as u32;
    let year = 100 * b + d - 4800 + m / 10;

    (year, month, day)
}

fn format_macaddr(bytes: &[u8; 6]) -> String {
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]
    )
}

fn format_inet4(bytes: &[u8; 4]) -> String {
    format!("{}.{}.{}.{}", bytes[0], bytes[1], bytes[2], bytes[3])
}

fn format_inet6(bytes: &[u8; 16]) -> String {
    let parts: Vec<String> = (0..8)
        .map(|i| {
            let high = bytes[i * 2] as u16;
            let low = bytes[i * 2 + 1] as u16;
            format!("{:x}", (high << 8) | low)
        })
        .collect();
    parts.join(":")
}

fn format_decimal(value: i128, scale: i16) -> String {
    if scale <= 0 {
        format!("{}", value * 10i128.pow((-scale) as u32))
    } else {
        let divisor = 10i128.pow(scale as u32);
        let int_part = value / divisor;
        let frac_part = (value % divisor).abs();
        format!("{}.{:0>width$}", int_part, frac_part, width = scale as usize)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        let mut result: String = s.chars().take(max_len - 3).collect();
        result.push_str("...");
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(values: Vec<OwnedValue>) -> Row {
        Row { values }
    }

    #[test]
    fn empty_table_renders_headers_only() {
        let headers = vec!["id".to_string(), "name".to_string()];
        let rows: Vec<Row> = vec![];

        let formatter = TableFormatter::new(headers, &rows);
        let output = formatter.render();

        assert!(output.contains("+----+------+"), "Should have separator line");
        assert!(output.contains("| id | name |"), "Should have header row");
        assert_eq!(formatter.row_count(), 0);
    }

    #[test]
    fn single_row_renders_correctly() {
        let headers = vec!["id".to_string(), "name".to_string()];
        let rows = vec![make_row(vec![
            OwnedValue::Int(1),
            OwnedValue::Text("Alice".to_string()),
        ])];

        let formatter = TableFormatter::new(headers, &rows);
        let output = formatter.render();

        assert!(output.contains("| 1  | Alice |"), "Should have data row with id=1, name=Alice");
        assert_eq!(formatter.row_count(), 1);
    }

    #[test]
    fn multiple_rows_render_correctly() {
        let headers = vec!["id".to_string(), "name".to_string()];
        let rows = vec![
            make_row(vec![OwnedValue::Int(1), OwnedValue::Text("Alice".to_string())]),
            make_row(vec![OwnedValue::Int(2), OwnedValue::Text("Bob".to_string())]),
        ];

        let formatter = TableFormatter::new(headers, &rows);
        let output = formatter.render();

        assert!(output.contains("| 1  | Alice |"));
        assert!(output.contains("| 2  | Bob   |"));
        assert_eq!(formatter.row_count(), 2);
    }

    #[test]
    fn null_value_displays_as_null() {
        let headers = vec!["value".to_string()];
        let rows = vec![make_row(vec![OwnedValue::Null])];

        let formatter = TableFormatter::new(headers, &rows);
        let output = formatter.render();

        assert!(output.contains("| NULL  |"), "NULL should be displayed as 'NULL'");
    }

    #[test]
    fn boolean_values_display_correctly() {
        let headers = vec!["flag".to_string()];
        let rows = vec![
            make_row(vec![OwnedValue::Bool(true)]),
            make_row(vec![OwnedValue::Bool(false)]),
        ];

        let formatter = TableFormatter::new(headers, &rows);
        let output = formatter.render();

        assert!(output.contains("true"), "true should be displayed");
        assert!(output.contains("false"), "false should be displayed");
    }

    #[test]
    fn float_values_format_without_trailing_zeros() {
        let formatted = format_value(&OwnedValue::Float(3.5));
        assert_eq!(formatted, "3.5");

        let formatted = format_value(&OwnedValue::Float(3.0));
        assert_eq!(formatted, "3");

        let formatted = format_value(&OwnedValue::Float(1.23456));
        assert_eq!(formatted, "1.23456");
    }

    #[test]
    fn blob_displays_as_hex() {
        let formatted = format_value(&OwnedValue::Blob(vec![0xDE, 0xAD, 0xBE, 0xEF]));
        assert_eq!(formatted, "x'DEADBEEF'");
    }

    #[test]
    fn long_blob_is_truncated() {
        let long_blob: Vec<u8> = (0..32).collect();
        let formatted = format_value(&OwnedValue::Blob(long_blob));

        assert!(formatted.starts_with("x'"));
        assert!(formatted.contains("..."));
        assert!(formatted.contains("32 bytes"));
    }

    #[test]
    fn uuid_formats_with_dashes() {
        let uuid_bytes: [u8; 16] = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4,
            0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00, 0x00,
        ];
        let formatted = format_value(&OwnedValue::Uuid(uuid_bytes));
        assert_eq!(formatted, "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn small_vector_shows_values() {
        let vec = vec![1.0f32, 2.0, 3.0];
        let formatted = format_value(&OwnedValue::Vector(vec));
        assert_eq!(formatted, "[1.000, 2.000, 3.000]");
    }

    #[test]
    fn large_vector_shows_summary() {
        let vec: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let formatted = format_value(&OwnedValue::Vector(vec));
        assert_eq!(formatted, "[f32; 128]");
    }

    #[test]
    fn long_text_is_truncated_at_max_width() {
        let truncated = truncate("This is a very long string that exceeds the maximum width", 20);
        assert_eq!(truncated.len(), 20);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn short_text_is_not_truncated() {
        let truncated = truncate("Short", 20);
        assert_eq!(truncated, "Short");
    }

    #[test]
    fn column_width_based_on_longest_value() {
        let headers = vec!["x".to_string()];
        let rows = vec![
            make_row(vec![OwnedValue::Text("short".to_string())]),
            make_row(vec![OwnedValue::Text("this is longer".to_string())]),
            make_row(vec![OwnedValue::Text("medium".to_string())]),
        ];

        let formatter = TableFormatter::new(headers, &rows);

        assert_eq!(formatter.widths[0], 14, "Width should be based on longest value");
    }

    #[test]
    fn interval_formats_correctly() {
        let formatted = format_interval(3_600_000_000, 0, 0);
        assert_eq!(formatted, "01:00:00");

        let formatted = format_interval(0, 5, 27);
        assert_eq!(formatted, "2 years 3 mons 5 days");

        let formatted = format_interval(0, 0, 0);
        assert_eq!(formatted, "00:00:00");
    }

    #[test]
    fn time_formats_correctly() {
        let formatted = format_time(3_600_000_000);
        assert_eq!(formatted, "01:00:00");

        let formatted = format_time(45_296_000_000);
        assert_eq!(formatted, "12:34:56");
    }
}
