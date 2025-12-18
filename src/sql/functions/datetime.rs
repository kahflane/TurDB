//! # DateTime Functions Module
//!
//! This module provides date and time SQL functions including:
//!
//! ## Current Date/Time
//! - `NOW()`, `CURRENT_TIMESTAMP` - Current datetime
//! - `CURRENT_DATE` - Current date only
//! - `CURRENT_TIME` - Current time only
//!
//! ## Date Formatting
//! - `DATE_FORMAT(format, datetime)` - Format datetime with pattern
//! - `STRFTIME(format, datetime)` - SQLite-style alias for DATE_FORMAT
//!
//! ## Date Arithmetic
//! - `DATE_ADD(date, days)` - Add days to date
//! - `DATE_SUB(date, days)` - Subtract days from date
//! - `DATEDIFF(date1, date2)` - Days between two dates
//!
//! ## Date Conversion
//! - `TO_DAYS(date)` - Convert date to day number (days since year 0)
//! - `FROM_DAYS(days)` - Convert day number to date
//! - `LAST_DAY(date)` - Get last day of the month
//!
//! ## Format Specifiers (DATE_FORMAT)
//!
//! | Specifier | Description | Example |
//! |-----------|-------------|---------|
//! | %Y | 4-digit year | 2024 |
//! | %y | 2-digit year | 24 |
//! | %m | Month (01-12) | 06 |
//! | %d | Day (01-31) | 15 |
//! | %H | Hour 24h (00-23) | 14 |
//! | %h | Hour 12h (01-12) | 02 |
//! | %i | Minutes (00-59) | 30 |
//! | %s | Seconds (00-59) | 45 |
//! | %p | AM/PM | PM |
//! | %M | Month name | June |
//! | %b | Month abbrev | Jun |
//! | %D | Day with suffix | 15th |
//! | %T | Time (HH:MM:SS) | 14:30:45 |
//! | %t | Last day of month | 30 |

use crate::types::Value;
use std::borrow::Cow;

/// Evaluates datetime functions by name.
pub fn eval_datetime_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "NOW" | "CURRENT_TIMESTAMP" => eval_now(),
        "CURRENT_DATE" => eval_current_date(),
        "CURRENT_TIME" => eval_current_time(),
        "DATE_FORMAT" | "STRFTIME" => eval_date_format(args),
        "DATE_ADD" => eval_date_add(args),
        "DATE_SUB" => eval_date_sub(args),
        "DATEDIFF" => eval_datediff(args),
        "TO_DAYS" => eval_to_days(args),
        "FROM_DAYS" => eval_from_days(args),
        "LAST_DAY" => eval_last_day(args),
        _ => None,
    }
}

fn eval_now<'a>() -> Option<Value<'a>> {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?;
    let secs = now.as_secs();
    let datetime = format_unix_timestamp_local(secs as i64);
    Some(Value::Text(Cow::Owned(datetime)))
}

fn eval_current_date<'a>() -> Option<Value<'a>> {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?;
    let secs = now.as_secs();
    let date = format_unix_date_local(secs as i64);
    Some(Value::Text(Cow::Owned(date)))
}

fn eval_current_time<'a>() -> Option<Value<'a>> {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?;
    let secs = now.as_secs();
    let time = format_unix_time_local(secs as i64);
    Some(Value::Text(Cow::Owned(time)))
}

fn eval_date_format<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
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

fn eval_date_add<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let date_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let days = match args.get(1)?.as_ref()? {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (year, month, day) = parse_date(&date_str)?;
    let day_number = date_to_days(year, month, day);
    let new_day_number = day_number + days;
    let (new_year, new_month, new_day) = days_to_date(new_day_number);
    
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        new_year, new_month, new_day
    ))))
}

fn eval_date_sub<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let date_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let days = match args.get(1)?.as_ref()? {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (year, month, day) = parse_date(&date_str)?;
    let day_number = date_to_days(year, month, day);
    let new_day_number = day_number - days;
    let (new_year, new_month, new_day) = days_to_date(new_day_number);
    
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        new_year, new_month, new_day
    ))))
}

fn eval_datediff<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let date1_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let date2_str = match args.get(1)?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (y1, m1, d1) = parse_date(&date1_str)?;
    let (y2, m2, d2) = parse_date(&date2_str)?;
    
    let days1 = date_to_days(y1, m1, d1);
    let days2 = date_to_days(y2, m2, d2);
    
    Some(Value::Int(days1 - days2))
}

fn eval_to_days<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (year, month, day) = parse_date(&date_str)?;
    let days = date_to_days(year, month, day);
    
    Some(Value::Int(days))
}

fn eval_from_days<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let days = match args.first()?.as_ref()? {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (year, month, day) = days_to_date(days);
    
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        year, month, day
    ))))
}

fn eval_last_day<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let (year, month, _) = parse_date(&date_str)?;
    let last = days_in_month(year, month);
    
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        year, month, last
    ))))
}

fn parse_date(s: &str) -> Option<(i64, u32, u32)> {
    let date_part = s.split(' ').next()?;
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() < 3 {
        return None;
    }
    let year: i64 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    Some((year, month, day))
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 30,
    }
}

fn date_to_days(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 12 } else { month };
    
    let days = 365 * y + y / 4 - y / 100 + y / 400
        + (153 * (m as i64 - 3) + 2) / 5
        + day as i64
        - 306;
    
    days
}

fn days_to_date(days: i64) -> (i64, u32, u32) {
    let z = days + 306;
    let h = 100 * z - 25;
    let a = h / 3652425;
    let b = a - a / 4;
    let y = (100 * b + h) / 36525;
    let c = b + z - 365 * y - y / 4;
    let m = (5 * c + 456) / 153;
    let d = c - (153 * m - 457) / 5;
    
    let (year, month) = if m > 12 {
        (y + 1, m - 12)
    } else {
        (y, m)
    };
    
    (year, month as u32, d as u32)
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
            // SAFETY: localtime_r is thread-safe and tm is properly initialized
            // before being read. The time_t value is valid (from SystemTime).
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

    let year_num: i64 = year.parse().unwrap_or(0);
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

    let last_day = days_in_month(year_num, month_num);

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
        .replace("%t", &last_day.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_to_days_and_back() {
        let cases = [
            (2017, 6, 15),
            (2000, 1, 1),
            (1970, 1, 1),
            (2024, 2, 29),
            (2023, 12, 31),
        ];
        
        for (year, month, day) in cases {
            let days = date_to_days(year, month, day);
            let (y, m, d) = days_to_date(days);
            assert_eq!((y, m, d), (year, month, day), "Round trip failed for {:04}-{:02}-{:02}", year, month, day);
        }
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2000, 2), 29);
        assert_eq!(days_in_month(1900, 2), 28);
        assert_eq!(days_in_month(2024, 1), 31);
        assert_eq!(days_in_month(2024, 4), 30);
    }

    #[test]
    fn test_date_add() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("2017-06-15"))),
            Some(Value::Int(10)),
        ];
        let result = eval_date_add(&args);
        assert_eq!(result, Some(Value::Text(Cow::Owned("2017-06-25".to_string()))));
    }

    #[test]
    fn test_date_sub() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("2017-06-15"))),
            Some(Value::Int(10)),
        ];
        let result = eval_date_sub(&args);
        assert_eq!(result, Some(Value::Text(Cow::Owned("2017-06-05".to_string()))));
    }

    #[test]
    fn test_datediff() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("2017-06-25"))),
            Some(Value::Text(Cow::Borrowed("2017-06-15"))),
        ];
        let result = eval_datediff(&args);
        assert_eq!(result, Some(Value::Int(10)));
    }

    #[test]
    fn test_last_day() {
        let args = vec![Some(Value::Text(Cow::Borrowed("2017-06-15")))];
        let result = eval_last_day(&args);
        assert_eq!(result, Some(Value::Text(Cow::Owned("2017-06-30".to_string()))));

        let args = vec![Some(Value::Text(Cow::Borrowed("2024-02-15")))];
        let result = eval_last_day(&args);
        assert_eq!(result, Some(Value::Text(Cow::Owned("2024-02-29".to_string()))));
    }

    #[test]
    fn test_to_days_from_days() {
        let args = vec![Some(Value::Text(Cow::Borrowed("2017-06-20")))];
        let days_result = eval_to_days(&args);
        
        if let Some(Value::Int(days)) = days_result {
            let from_args = vec![Some(Value::Int(days))];
            let date_result = eval_from_days(&from_args);
            assert_eq!(date_result, Some(Value::Text(Cow::Owned("2017-06-20".to_string()))));
        } else {
            panic!("TO_DAYS should return an integer");
        }
    }
}
