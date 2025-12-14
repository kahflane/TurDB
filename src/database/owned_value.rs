use crate::records::types::DataType;
use crate::records::RecordView;
use crate::types::Value;
use eyre::Result;
use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Vector(Vec<f32>),
    Date(i32),
    Time(i64),
    Timestamp(i64),
    TimestampTz(i64, i32),
    Uuid([u8; 16]),
    MacAddr([u8; 6]),
    Inet4([u8; 4]),
    Inet6([u8; 16]),
    Interval(i64, i32, i32),
    Point(f64, f64),
    Box((f64, f64), (f64, f64)),
    Circle((f64, f64), f64),
    Jsonb(Vec<u8>),
}

impl<'a> From<&Value<'a>> for OwnedValue {
    fn from(v: &Value<'a>) -> Self {
        match v {
            Value::Null => OwnedValue::Null,
            Value::Int(i) => OwnedValue::Int(*i),
            Value::Float(f) => OwnedValue::Float(*f),
            Value::Text(s) => OwnedValue::Text(s.to_string()),
            Value::Blob(b) => OwnedValue::Blob(b.to_vec()),
            Value::Vector(v) => OwnedValue::Vector(v.to_vec()),
            Value::Uuid(u) => OwnedValue::Uuid(*u),
            Value::Jsonb(b) => OwnedValue::Jsonb(b.to_vec()),
        }
    }
}

impl OwnedValue {
    pub fn to_value(&self) -> Value<'_> {
        match self {
            OwnedValue::Null => Value::Null,
            OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
            OwnedValue::Int(i) => Value::Int(*i),
            OwnedValue::Float(f) => Value::Float(*f),
            OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
            OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
            OwnedValue::Vector(v) => Value::Vector(Cow::Borrowed(v.as_slice())),
            OwnedValue::Date(d) => Value::Int(*d as i64),
            OwnedValue::Time(t) => Value::Int(*t),
            OwnedValue::Timestamp(ts) => Value::Int(*ts),
            OwnedValue::TimestampTz(ts, _tz) => Value::Int(*ts),
            OwnedValue::Uuid(u) => Value::Uuid(*u),
            OwnedValue::MacAddr(m) => Value::Blob(Cow::Borrowed(m.as_slice())),
            OwnedValue::Inet4(ip) => Value::Blob(Cow::Borrowed(ip.as_slice())),
            OwnedValue::Inet6(ip) => Value::Blob(Cow::Borrowed(ip.as_slice())),
            OwnedValue::Interval(micros, _days, _months) => Value::Int(*micros),
            OwnedValue::Point(x, y) => Value::Text(Cow::Owned(format!("({},{})", x, y))),
            OwnedValue::Box(p1, p2) => {
                Value::Text(Cow::Owned(format!("(({},{}),({},{}))", p1.0, p1.1, p2.0, p2.1)))
            }
            OwnedValue::Circle(center, radius) => {
                Value::Text(Cow::Owned(format!("<({},{}),{}>", center.0, center.1, radius)))
            }
            OwnedValue::Jsonb(data) => Value::Jsonb(Cow::Borrowed(data.as_slice())),
        }
    }

    pub fn from_record_column(
        record: &RecordView<'_>,
        col_idx: usize,
        data_type: DataType,
    ) -> Result<Self> {
        let owned_val = match data_type {
            DataType::Int8 => record
                .get_int8_opt(col_idx)?
                .map(OwnedValue::Int)
                .unwrap_or(OwnedValue::Null),
            DataType::Int4 => record
                .get_int4_opt(col_idx)?
                .map(|i| OwnedValue::Int(i as i64))
                .unwrap_or(OwnedValue::Null),
            DataType::Int2 => record
                .get_int2_opt(col_idx)?
                .map(|i| OwnedValue::Int(i as i64))
                .unwrap_or(OwnedValue::Null),
            DataType::Float8 => record
                .get_float8_opt(col_idx)?
                .map(OwnedValue::Float)
                .unwrap_or(OwnedValue::Null),
            DataType::Float4 => record
                .get_float4_opt(col_idx)?
                .map(|f| OwnedValue::Float(f as f64))
                .unwrap_or(OwnedValue::Null),
            DataType::Text => record
                .get_text_opt(col_idx)?
                .map(|s| OwnedValue::Text(s.to_string()))
                .unwrap_or(OwnedValue::Null),
            DataType::Blob => record
                .get_blob_opt(col_idx)?
                .map(|b| OwnedValue::Blob(b.to_vec()))
                .unwrap_or(OwnedValue::Null),
            DataType::Bool => record
                .get_bool_opt(col_idx)?
                .map(OwnedValue::Bool)
                .unwrap_or(OwnedValue::Null),
            DataType::Date => record
                .get_date_opt(col_idx)?
                .map(OwnedValue::Date)
                .unwrap_or(OwnedValue::Null),
            DataType::Time => record
                .get_time_opt(col_idx)?
                .map(OwnedValue::Time)
                .unwrap_or(OwnedValue::Null),
            DataType::Timestamp => record
                .get_timestamp_opt(col_idx)?
                .map(OwnedValue::Timestamp)
                .unwrap_or(OwnedValue::Null),
            DataType::TimestampTz => record
                .get_timestamptz_opt(col_idx)?
                .map(|(ts, tz)| OwnedValue::TimestampTz(ts, tz))
                .unwrap_or(OwnedValue::Null),
            DataType::Uuid => record
                .get_uuid_opt(col_idx)?
                .map(|u| OwnedValue::Uuid(*u))
                .unwrap_or(OwnedValue::Null),
            DataType::MacAddr => record
                .get_macaddr_opt(col_idx)?
                .map(|m| OwnedValue::MacAddr(*m))
                .unwrap_or(OwnedValue::Null),
            DataType::Inet4 => record
                .get_inet4_opt(col_idx)?
                .map(|ip| OwnedValue::Inet4(*ip))
                .unwrap_or(OwnedValue::Null),
            DataType::Inet6 => record
                .get_inet6_opt(col_idx)?
                .map(|ip| OwnedValue::Inet6(*ip))
                .unwrap_or(OwnedValue::Null),
            DataType::Interval => record
                .get_interval_opt(col_idx)?
                .map(|(micros, days, months)| OwnedValue::Interval(micros, days, months))
                .unwrap_or(OwnedValue::Null),
            DataType::Point => record
                .get_point_opt(col_idx)?
                .map(|(x, y)| OwnedValue::Point(x, y))
                .unwrap_or(OwnedValue::Null),
            DataType::Box => record
                .get_box_opt(col_idx)?
                .map(|(p1, p2)| OwnedValue::Box(p1, p2))
                .unwrap_or(OwnedValue::Null),
            DataType::Circle => record
                .get_circle_opt(col_idx)?
                .map(|(center, radius)| OwnedValue::Circle(center, radius))
                .unwrap_or(OwnedValue::Null),
            DataType::Vector => record
                .get_vector_opt(col_idx)?
                .map(OwnedValue::Vector)
                .unwrap_or(OwnedValue::Null),
            DataType::Jsonb => record
                .get_jsonb_opt(col_idx)?
                .map(|v| OwnedValue::Jsonb(v.data().to_vec()))
                .unwrap_or(OwnedValue::Null),
            DataType::Decimal => record
                .get_decimal_opt(col_idx)?
                .map(|d| OwnedValue::Text(format!("{}", d.digits())))
                .unwrap_or(OwnedValue::Null),
            DataType::Int4Range | DataType::Int8Range | DataType::DateRange | DataType::TimestampRange => {
                record
                    .get_blob_opt(col_idx)?
                    .map(|b| OwnedValue::Blob(b.to_vec()))
                    .unwrap_or(OwnedValue::Null)
            }
            DataType::Enum => record
                .get_enum_opt(col_idx)?
                .map(|(type_id, variant)| OwnedValue::Int(((type_id as i64) << 16) | (variant as i64)))
                .unwrap_or(OwnedValue::Null),
            DataType::Composite | DataType::Array => record
                .get_blob_opt(col_idx)?
                .map(|b| OwnedValue::Blob(b.to_vec()))
                .unwrap_or(OwnedValue::Null),
        };
        Ok(owned_val)
    }

    pub fn extract_row_from_record(
        record: &RecordView<'_>,
        columns: &[crate::schema::ColumnDef],
    ) -> Result<Vec<Self>> {
        let mut row_values = Vec::with_capacity(columns.len());
        for (col_idx, col_def) in columns.iter().enumerate() {
            let owned_val = Self::from_record_column(record, col_idx, col_def.data_type())?;
            row_values.push(owned_val);
        }
        Ok(row_values)
    }
}

pub fn owned_values_to_values(owned: &[OwnedValue]) -> Vec<Value<'_>> {
    owned.iter().map(|ov| ov.to_value()).collect()
}

pub fn create_record_schema(columns: &[crate::schema::ColumnDef]) -> crate::records::Schema {
    use crate::records::types::ColumnDef as RecordColumnDef;
    let record_columns: Vec<RecordColumnDef> = columns
        .iter()
        .map(|c| RecordColumnDef::new(c.name().to_string(), c.data_type()))
        .collect();
    crate::records::Schema::new(record_columns)
}

pub fn create_column_map(columns: &[crate::schema::ColumnDef]) -> Vec<(String, usize)> {
    columns
        .iter()
        .enumerate()
        .map(|(idx, col)| (col.name().to_lowercase(), idx))
        .collect()
}

impl OwnedValue {
    pub fn set_in_builder(
        &self,
        builder: &mut crate::records::RecordBuilder<'_>,
        idx: usize,
    ) -> Result<()> {
        match self {
            OwnedValue::Null => builder.set_null(idx),
            OwnedValue::Bool(b) => builder.set_bool(idx, *b)?,
            OwnedValue::Int(i) => builder.set_int8(idx, *i)?,
            OwnedValue::Float(f) => builder.set_float8(idx, *f)?,
            OwnedValue::Text(s) => builder.set_text(idx, s)?,
            OwnedValue::Blob(b) => builder.set_blob(idx, b)?,
            OwnedValue::Vector(v) => builder.set_vector(idx, v)?,
            OwnedValue::Date(d) => builder.set_date(idx, *d)?,
            OwnedValue::Time(t) => builder.set_time(idx, *t)?,
            OwnedValue::Timestamp(ts) => builder.set_timestamp(idx, *ts)?,
            OwnedValue::TimestampTz(ts, tz) => builder.set_timestamptz(idx, *ts, *tz)?,
            OwnedValue::Uuid(u) => builder.set_uuid(idx, u)?,
            OwnedValue::MacAddr(m) => builder.set_macaddr(idx, m)?,
            OwnedValue::Inet4(ip) => builder.set_inet4(idx, ip)?,
            OwnedValue::Inet6(ip) => builder.set_inet6(idx, ip)?,
            OwnedValue::Interval(micros, days, months) => {
                builder.set_interval(idx, *micros, *days, *months)?
            }
            OwnedValue::Point(x, y) => builder.set_point(idx, *x, *y)?,
            OwnedValue::Box(p1, p2) => builder.set_box(idx, *p1, *p2)?,
            OwnedValue::Circle(center, radius) => builder.set_circle(idx, *center, *radius)?,
            OwnedValue::Jsonb(data) => builder.set_jsonb_bytes(idx, data)?,
        }
        Ok(())
    }

    pub fn build_record_from_values(
        values: &[OwnedValue],
        schema: &crate::records::Schema,
    ) -> Result<Vec<u8>> {
        let mut builder = crate::records::RecordBuilder::new(schema);
        for (idx, val) in values.iter().enumerate() {
            val.set_in_builder(&mut builder, idx)?;
        }
        builder.build()
    }

    pub fn jsonb_get(&self, key: &str) -> Result<Option<OwnedValue>> {
        use crate::records::jsonb::JsonbView;

        match self {
            OwnedValue::Jsonb(data) => {
                let view = JsonbView::new(data)?;
                match view.get(key)? {
                    Some(val) => Ok(Some(Self::from_jsonb_value(&val)?)),
                    None => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    pub fn jsonb_get_path(&self, path: &[&str]) -> Result<Option<OwnedValue>> {
        use crate::records::jsonb::JsonbView;

        match self {
            OwnedValue::Jsonb(data) => {
                let view = JsonbView::new(data)?;
                match view.get_path(path)? {
                    Some(val) => Ok(Some(Self::from_jsonb_value(&val)?)),
                    None => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    pub fn jsonb_array_get(&self, idx: usize) -> Result<Option<OwnedValue>> {
        use crate::records::jsonb::JsonbView;

        match self {
            OwnedValue::Jsonb(data) => {
                let view = JsonbView::new(data)?;
                match view.array_get(idx)? {
                    Some(val) => Ok(Some(Self::from_jsonb_value(&val)?)),
                    None => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    fn from_jsonb_value(val: &crate::records::jsonb::JsonbValue<'_>) -> Result<OwnedValue> {
        use crate::records::jsonb::JsonbValue;

        Ok(match val {
            JsonbValue::Null => OwnedValue::Null,
            JsonbValue::Bool(b) => OwnedValue::Bool(*b),
            JsonbValue::Number(n) => OwnedValue::Float(*n),
            JsonbValue::String(s) => OwnedValue::Text(s.to_string()),
            JsonbValue::Array(view) => OwnedValue::Jsonb(view.data().to_vec()),
            JsonbValue::Object(view) => OwnedValue::Jsonb(view.data().to_vec()),
        })
    }

    pub fn is_jsonb(&self) -> bool {
        matches!(self, OwnedValue::Jsonb(_))
    }
}