//! # DateTime Functions Module
//!
//! This module provides date and time SQL functions.
//!
//! ## Current Date/Time
//! - `NOW()`, `CURRENT_TIMESTAMP`, `LOCALTIME`, `LOCALTIMESTAMP`, `SYSDATE()` - Current datetime
//! - `CURDATE()`, `CURRENT_DATE` - Current date
//! - `CURTIME()`, `CURRENT_TIME` - Current time
//!
//! ## Date Extraction
//! - `DATE(datetime)` - Extract date part
//! - `TIME(datetime)` - Extract time part
//! - `YEAR(date)`, `MONTH(date)`, `DAY(date)` / `DAYOFMONTH(date)`
//! - `HOUR(time)`, `MINUTE(time)`, `SECOND(time)`, `MICROSECOND(time)`
//! - `DAYNAME(date)`, `MONTHNAME(date)` - Name of day/month
//! - `DAYOFWEEK(date)`, `DAYOFYEAR(date)`, `WEEKDAY(date)`
//! - `WEEK(date)`, `WEEKOFYEAR(date)`, `YEARWEEK(date)`
//! - `QUARTER(date)` - Quarter (1-4)
//! - `EXTRACT(unit FROM datetime)` - Extract component
//!
//! ## Date Arithmetic
//! - `DATE_ADD(date, days)` / `ADDDATE(date, days)` - Add days
//! - `DATE_SUB(date, days)` / `SUBDATE(date, days)` - Subtract days
//! - `ADDTIME(datetime, time)` - Add time
//! - `SUBTIME(datetime, time)` - Subtract time
//! - `DATEDIFF(date1, date2)` - Days between dates
//! - `TIMEDIFF(time1, time2)` - Time difference
//!
//! ## Date Conversion
//! - `TO_DAYS(date)` - Date to day number
//! - `FROM_DAYS(n)` - Day number to date
//! - `TIME_TO_SEC(time)` - Time to seconds
//! - `SEC_TO_TIME(secs)` - Seconds to time
//! - `MAKEDATE(year, dayofyear)` - Create date
//! - `MAKETIME(hour, minute, second)` - Create time
//! - `TIMESTAMP(date, time)` - Combine date and time
//! - `STR_TO_DATE(str, format)` - Parse date string
//!
//! ## Date Info
//! - `LAST_DAY(date)` - Last day of month
//! - `PERIOD_ADD(period, n)` - Add months to period (YYYYMM)
//! - `PERIOD_DIFF(p1, p2)` - Months between periods
//!
//! ## Formatting
//! - `DATE_FORMAT(format, datetime)` / `STRFTIME(format, datetime)`
//! - `TIME_FORMAT(format, time)`

use crate::types::Value;
use std::borrow::Cow;

pub fn eval_datetime_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "NOW" | "CURRENT_TIMESTAMP" | "LOCALTIME" | "LOCALTIMESTAMP" | "SYSDATE" => eval_now(),
        "CURDATE" | "CURRENT_DATE" => eval_current_date(),
        "CURTIME" | "CURRENT_TIME" => eval_current_time(),
        "DATE" => eval_date(args),
        "TIME" => eval_time(args),
        "YEAR" => eval_year(args),
        "MONTH" => eval_month(args),
        "DAY" | "DAYOFMONTH" => eval_day(args),
        "HOUR" => eval_hour(args),
        "MINUTE" => eval_minute(args),
        "SECOND" => eval_second(args),
        "MICROSECOND" => eval_microsecond(args),
        "DAYNAME" => eval_dayname(args),
        "MONTHNAME" => eval_monthname(args),
        "DAYOFWEEK" => eval_dayofweek(args),
        "DAYOFYEAR" => eval_dayofyear(args),
        "WEEKDAY" => eval_weekday(args),
        "WEEK" | "WEEKOFYEAR" => eval_week(args),
        "YEARWEEK" => eval_yearweek(args),
        "QUARTER" => eval_quarter(args),
        "DATE_ADD" | "ADDDATE" => eval_date_add(args),
        "DATE_SUB" | "SUBDATE" => eval_date_sub(args),
        "ADDTIME" => eval_addtime(args),
        "SUBTIME" => eval_subtime(args),
        "DATEDIFF" => eval_datediff(args),
        "TIMEDIFF" => eval_timediff(args),
        "TO_DAYS" => eval_to_days(args),
        "FROM_DAYS" => eval_from_days(args),
        "TIME_TO_SEC" => eval_time_to_sec(args),
        "SEC_TO_TIME" => eval_sec_to_time(args),
        "MAKEDATE" => eval_makedate(args),
        "MAKETIME" => eval_maketime(args),
        "TIMESTAMP" => eval_timestamp(args),
        "LAST_DAY" => eval_last_day(args),
        "PERIOD_ADD" => eval_period_add(args),
        "PERIOD_DIFF" => eval_period_diff(args),
        "DATE_FORMAT" | "STRFTIME" => eval_date_format(args),
        "TIME_FORMAT" => eval_time_format(args),
        "STR_TO_DATE" => eval_str_to_date(args),
        _ => None,
    }
}

fn get_text<'a>(val: &Option<Value<'a>>) -> Option<String> {
    match val.as_ref()? {
        Value::Text(s) => Some(s.to_string()),
        Value::Int(n) => Some(n.to_string()),
        Value::Null => None,
        _ => None,
    }
}

fn get_int(val: &Option<Value>) -> Option<i64> {
    match val.as_ref()? {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        Value::Text(s) => s.parse().ok(),
        Value::Null => None,
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

fn eval_date<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let date = datetime.split(' ').next()?.to_string();
    Some(Value::Text(Cow::Owned(date)))
}

fn eval_time<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let time = datetime.split(' ').nth(1).unwrap_or("00:00:00").to_string();
    Some(Value::Text(Cow::Owned(time)))
}

fn eval_year<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, _, _) = parse_date(&datetime)?;
    Some(Value::Int(year))
}

fn eval_month<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, month, _) = parse_date(&datetime)?;
    Some(Value::Int(month as i64))
}

fn eval_day<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, _, day) = parse_date(&datetime)?;
    Some(Value::Int(day as i64))
}

fn eval_hour<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (hour, _, _) = parse_time(&datetime)?;
    Some(Value::Int(hour as i64))
}

fn eval_minute<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, minute, _) = parse_time(&datetime)?;
    Some(Value::Int(minute as i64))
}

fn eval_second<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, _, second) = parse_time(&datetime)?;
    Some(Value::Int(second as i64))
}

fn eval_microsecond<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let time_part = datetime.split(' ').nth(1).unwrap_or(&datetime);
    if let Some(dot_pos) = time_part.find('.') {
        let micros: i64 = time_part[dot_pos + 1..].parse().unwrap_or(0);
        Some(Value::Int(micros))
    } else {
        Some(Value::Int(0))
    }
}

fn eval_dayname<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let dow = day_of_week(year, month, day);
    let names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    Some(Value::Text(Cow::Borrowed(names[dow as usize])))
}

fn eval_monthname<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, month, _) = parse_date(&datetime)?;
    let names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ];
    Some(Value::Text(Cow::Borrowed(names[(month - 1) as usize])))
}

fn eval_dayofweek<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let dow = day_of_week(year, month, day);
    Some(Value::Int(dow as i64 + 1))
}

fn eval_dayofyear<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let doy = day_of_year(year, month, day);
    Some(Value::Int(doy as i64))
}

fn eval_weekday<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let dow = day_of_week(year, month, day);
    let weekday = if dow == 0 { 6 } else { dow - 1 };
    Some(Value::Int(weekday as i64))
}

fn eval_week<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let doy = day_of_year(year, month, day);
    let week = (doy + 6) / 7;
    Some(Value::Int(week as i64))
}

fn eval_yearweek<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&datetime)?;
    let doy = day_of_year(year, month, day);
    let week = (doy + 6) / 7;
    Some(Value::Int(year * 100 + week as i64))
}

fn eval_quarter<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let (_, month, _) = parse_date(&datetime)?;
    let quarter = (month - 1) / 3 + 1;
    Some(Value::Int(quarter as i64))
}

fn eval_date_add<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = get_text(args.first()?)?;
    let days = get_int(args.get(1)?)?;
    
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
    let date_str = get_text(args.first()?)?;
    let days = get_int(args.get(1)?)?;
    
    let (year, month, day) = parse_date(&date_str)?;
    let day_number = date_to_days(year, month, day);
    let new_day_number = day_number - days;
    let (new_year, new_month, new_day) = days_to_date(new_day_number);
    
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        new_year, new_month, new_day
    ))))
}

fn eval_addtime<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let time_to_add = get_text(args.get(1)?)?;
    
    let (hour1, min1, sec1) = parse_time(&datetime)?;
    let (hour2, min2, sec2) = parse_time(&time_to_add)?;
    
    let total_secs = (hour1 as i64 * 3600 + min1 as i64 * 60 + sec1 as i64)
        + (hour2 as i64 * 3600 + min2 as i64 * 60 + sec2 as i64);
    
    let new_hour = (total_secs / 3600) % 24;
    let new_min = (total_secs % 3600) / 60;
    let new_sec = total_secs % 60;
    
    let date_part = datetime.split(' ').next().unwrap_or("");
    if date_part.contains('-') {
        Some(Value::Text(Cow::Owned(format!(
            "{} {:02}:{:02}:{:02}",
            date_part, new_hour, new_min, new_sec
        ))))
    } else {
        Some(Value::Text(Cow::Owned(format!(
            "{:02}:{:02}:{:02}",
            new_hour, new_min, new_sec
        ))))
    }
}

fn eval_subtime<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let datetime = get_text(args.first()?)?;
    let time_to_sub = get_text(args.get(1)?)?;
    
    let (hour1, min1, sec1) = parse_time(&datetime)?;
    let (hour2, min2, sec2) = parse_time(&time_to_sub)?;
    
    let total_secs = (hour1 as i64 * 3600 + min1 as i64 * 60 + sec1 as i64)
        - (hour2 as i64 * 3600 + min2 as i64 * 60 + sec2 as i64);
    
    let total_secs = if total_secs < 0 { total_secs + 86400 } else { total_secs };
    
    let new_hour = (total_secs / 3600) % 24;
    let new_min = (total_secs % 3600) / 60;
    let new_sec = total_secs % 60;
    
    let date_part = datetime.split(' ').next().unwrap_or("");
    if date_part.contains('-') {
        Some(Value::Text(Cow::Owned(format!(
            "{} {:02}:{:02}:{:02}",
            date_part, new_hour, new_min, new_sec
        ))))
    } else {
        Some(Value::Text(Cow::Owned(format!(
            "{:02}:{:02}:{:02}",
            new_hour, new_min, new_sec
        ))))
    }
}

fn eval_datediff<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date1_str = get_text(args.first()?)?;
    let date2_str = get_text(args.get(1)?)?;
    
    let (y1, m1, d1) = parse_date(&date1_str)?;
    let (y2, m2, d2) = parse_date(&date2_str)?;
    
    let days1 = date_to_days(y1, m1, d1);
    let days2 = date_to_days(y2, m2, d2);
    
    Some(Value::Int(days1 - days2))
}

fn eval_timediff<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let time1 = get_text(args.first()?)?;
    let time2 = get_text(args.get(1)?)?;
    
    let (h1, m1, s1) = parse_time(&time1)?;
    let (h2, m2, s2) = parse_time(&time2)?;
    
    let secs1 = h1 as i64 * 3600 + m1 as i64 * 60 + s1 as i64;
    let secs2 = h2 as i64 * 3600 + m2 as i64 * 60 + s2 as i64;
    let diff = secs1 - secs2;
    
    let sign = if diff < 0 { "-" } else { "" };
    let diff = diff.abs();
    let hours = diff / 3600;
    let mins = (diff % 3600) / 60;
    let secs = diff % 60;
    
    Some(Value::Text(Cow::Owned(format!(
        "{}{:02}:{:02}:{:02}",
        sign, hours, mins, secs
    ))))
}

fn eval_to_days<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = get_text(args.first()?)?;
    let (year, month, day) = parse_date(&date_str)?;
    let days = date_to_days(year, month, day);
    Some(Value::Int(days))
}

fn eval_from_days<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let days = get_int(args.first()?)?;
    let (year, month, day) = days_to_date(days);
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        year, month, day
    ))))
}

fn eval_time_to_sec<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let time = get_text(args.first()?)?;
    let (hour, min, sec) = parse_time(&time)?;
    let total = hour as i64 * 3600 + min as i64 * 60 + sec as i64;
    Some(Value::Int(total))
}

fn eval_sec_to_time<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let secs = get_int(args.first()?)?;
    let sign = if secs < 0 { "-" } else { "" };
    let secs = secs.abs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    Some(Value::Text(Cow::Owned(format!(
        "{}{:02}:{:02}:{:02}",
        sign, hours, mins, s
    ))))
}

fn eval_makedate<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let year = get_int(args.first()?)?;
    let dayofyear = get_int(args.get(1)?)?;
    
    if dayofyear < 1 {
        return Some(Value::Null);
    }
    
    let jan1_days = date_to_days(year, 1, 1);
    let target_days = jan1_days + dayofyear - 1;
    let (y, m, d) = days_to_date(target_days);
    
    Some(Value::Text(Cow::Owned(format!("{:04}-{:02}-{:02}", y, m, d))))
}

fn eval_maketime<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let hour = get_int(args.first()?)?;
    let minute = get_int(args.get(1)?)?;
    let second = get_int(args.get(2)?)?;
    
    Some(Value::Text(Cow::Owned(format!(
        "{:02}:{:02}:{:02}",
        hour, minute, second
    ))))
}

fn eval_timestamp<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date = get_text(args.first()?)?;
    let time = args.get(1).and_then(|v| get_text(v)).unwrap_or_else(|| "00:00:00".to_string());
    
    let date_part = date.split(' ').next().unwrap_or(&date);
    Some(Value::Text(Cow::Owned(format!("{} {}", date_part, time))))
}

fn eval_last_day<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = get_text(args.first()?)?;
    let (year, month, _) = parse_date(&date_str)?;
    let last = days_in_month(year, month);
    Some(Value::Text(Cow::Owned(format!(
        "{:04}-{:02}-{:02}",
        year, month, last
    ))))
}

fn eval_period_add<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let period = get_int(args.first()?)?;
    let months = get_int(args.get(1)?)?;
    
    let year = period / 100;
    let month = period % 100;
    
    let total_months = year * 12 + month + months - 1;
    let new_year = total_months / 12;
    let new_month = total_months % 12 + 1;
    
    Some(Value::Int(new_year * 100 + new_month))
}

fn eval_period_diff<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let p1 = get_int(args.first()?)?;
    let p2 = get_int(args.get(1)?)?;
    
    let months1 = (p1 / 100) * 12 + p1 % 100;
    let months2 = (p2 / 100) * 12 + p2 % 100;
    
    Some(Value::Int(months1 - months2))
}

fn eval_date_format<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let format_str = get_text(args.first()?)?;
    let datetime_str = get_text(args.get(1)?)?;
    let formatted = format_datetime_with_pattern(&datetime_str, &format_str);
    Some(Value::Text(Cow::Owned(formatted)))
}

fn eval_time_format<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let format_str = get_text(args.first()?)?;
    let time_str = get_text(args.get(1)?)?;
    let formatted = format_datetime_with_pattern(&time_str, &format_str);
    Some(Value::Text(Cow::Owned(formatted)))
}

fn eval_str_to_date<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let date_str = get_text(args.first()?)?;
    let _format = get_text(args.get(1)?)?;
    if parse_date(&date_str).is_some() {
        Some(Value::Text(Cow::Owned(date_str)))
    } else {
        Some(Value::Null)
    }
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

fn parse_time(s: &str) -> Option<(u32, u32, u32)> {
    let time_part = if s.contains(' ') {
        s.split(' ').nth(1)?
    } else {
        s
    };
    let time_without_micros = time_part.split('.').next()?;
    let parts: Vec<&str> = time_without_micros.split(':').collect();
    let hour: u32 = parts.first()?.parse().ok()?;
    let minute: u32 = parts.get(1).unwrap_or(&"0").parse().ok()?;
    let second: u32 = parts.get(2).unwrap_or(&"0").parse().ok()?;
    Some((hour, minute, second))
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
    365 * y + y / 4 - y / 100 + y / 400 + (153 * (m as i64 - 3) + 2) / 5 + day as i64 - 306
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
    let (year, month) = if m > 12 { (y + 1, m - 12) } else { (y, m) };
    (year, month as u32, d as u32)
}

fn day_of_week(year: i64, month: u32, day: u32) -> u32 {
    let (y, m) = if month < 3 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let q = day as i64;
    let k = y % 100;
    let j = y / 100;
    let h = (q + (13 * (m as i64 + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7;
    let dow = ((h + 6) % 7) as u32;
    dow
}

fn day_of_year(year: i64, month: u32, day: u32) -> u32 {
    let jan1 = date_to_days(year, 1, 1);
    let current = date_to_days(year, month, day);
    (current - jan1 + 1) as u32
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
            if libc::localtime_r(&time_t, tm.as_mut_ptr()).is_null() {
                return 0;
            }
            let tm = tm.assume_init();
            tm.tm_gmtoff as i64
        }
    }
    #[cfg(not(unix))]
    { 0 }
}

fn format_unix_timestamp_local(secs: i64) -> String {
    let offset = get_local_timezone_offset();
    format_unix_timestamp(secs + offset)
}

fn format_unix_timestamp(secs: i64) -> String {
    const SECONDS_PER_DAY: i64 = 86400;
    let mut days = secs / SECONDS_PER_DAY;
    let day_secs = secs % SECONDS_PER_DAY;
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;
    days += 719468;
    let era = if days >= 0 { days / 146097 } else { (days - 146096) / 146097 };
    let doe = (days - era * 146097) as u32;
    let yoe = (doe - doe / 1461 + doe / 36524 - doe / 146097) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, m, d, hours, minutes, seconds)
}

fn format_unix_date_local(secs: i64) -> String {
    format_unix_timestamp_local(secs).split(' ').next().unwrap_or("").to_string()
}

fn format_unix_time_local(secs: i64) -> String {
    format_unix_timestamp_local(secs).split(' ').nth(1).unwrap_or("").to_string()
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
    let hour_12 = if hour_num == 0 { 12 } else if hour_num > 12 { hour_num - 12 } else { hour_num };
    let am_pm = if hour_num < 12 { "AM" } else { "PM" };

    let year_num: i64 = year.parse().unwrap_or(0);
    let month_num: u32 = month.parse().unwrap_or(1);
    let day_num: u32 = day.parse().unwrap_or(1);
    
    let month_names = ["January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"];
    let month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let day_abbrs = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    
    let month_name = month_names.get((month_num as usize).saturating_sub(1)).unwrap_or(&"");
    let month_abbr = month_abbrs.get((month_num as usize).saturating_sub(1)).unwrap_or(&"");
    let dow = day_of_week(year_num, month_num, day_num) as usize;
    let day_name = day_names.get(dow).unwrap_or(&"");
    let day_abbr = day_abbrs.get(dow).unwrap_or(&"");
    let last_day = days_in_month(year_num, month_num);
    let doy = day_of_year(year_num, month_num, day_num);

    pattern
        .replace("%Y", year)
        .replace("%y", &year[year.len().saturating_sub(2)..])
        .replace("%m", month)
        .replace("%c", &month_num.to_string())
        .replace("%d", day)
        .replace("%e", &day_num.to_string())
        .replace("%H", hour)
        .replace("%k", &hour_num.to_string())
        .replace("%i", minute)
        .replace("%s", second)
        .replace("%S", second)
        .replace("%M", month_name)
        .replace("%b", month_abbr)
        .replace("%W", day_name)
        .replace("%a", day_abbr)
        .replace("%h", &format!("{:02}", hour_12))
        .replace("%I", &format!("{:02}", hour_12))
        .replace("%l", &hour_12.to_string())
        .replace("%p", am_pm)
        .replace("%T", &format!("{}:{}:{}", hour, minute, second))
        .replace("%r", &format!("{:02}:{:02}:{:02} {}", hour_12, minute, second, am_pm))
        .replace("%D", &format!("{}{}", day, ordinal_suffix(day_num)))
        .replace("%t", &last_day.to_string())
        .replace("%j", &format!("{:03}", doy))
        .replace("%w", &dow.to_string())
        .replace("%U", &format!("{:02}", (doy + 6) / 7))
        .replace("%%", "%")
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
    fn test_date_parts() {
        let args = vec![Some(Value::Text(Cow::Borrowed("2024-06-15 14:30:45")))];
        assert_eq!(eval_year(&args), Some(Value::Int(2024)));
        assert_eq!(eval_month(&args), Some(Value::Int(6)));
        assert_eq!(eval_day(&args), Some(Value::Int(15)));
        assert_eq!(eval_hour(&args), Some(Value::Int(14)));
        assert_eq!(eval_minute(&args), Some(Value::Int(30)));
        assert_eq!(eval_second(&args), Some(Value::Int(45)));
    }

    #[test]
    fn test_dayname_monthname() {
        let args = vec![Some(Value::Text(Cow::Borrowed("2024-06-15")))];
        assert_eq!(eval_dayname(&args), Some(Value::Text(Cow::Borrowed("Saturday"))));
        assert_eq!(eval_monthname(&args), Some(Value::Text(Cow::Borrowed("June"))));
    }

    #[test]
    fn test_makedate() {
        let args = vec![Some(Value::Int(2024)), Some(Value::Int(100))];
        assert_eq!(eval_makedate(&args), Some(Value::Text(Cow::Owned("2024-04-09".to_string()))));
    }

    #[test]
    fn test_time_to_sec() {
        let args = vec![Some(Value::Text(Cow::Borrowed("01:30:00")))];
        assert_eq!(eval_time_to_sec(&args), Some(Value::Int(5400)));
    }

    #[test]
    fn test_sec_to_time() {
        let args = vec![Some(Value::Int(5400))];
        assert_eq!(eval_sec_to_time(&args), Some(Value::Text(Cow::Owned("01:30:00".to_string()))));
    }
}
