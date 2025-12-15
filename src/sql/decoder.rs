use crate::types::Value;
use std::borrow::Cow;

pub trait RecordDecoder {
    fn decode(&self, key: &[u8], value: &[u8]) -> eyre::Result<Vec<Value<'static>>>;
}

pub struct SimpleDecoder {
    column_types: Vec<crate::records::types::DataType>,
    decode_columns: Vec<usize>,
    cached_schema: crate::records::Schema,
}

impl SimpleDecoder {
    pub fn new(column_types: Vec<crate::records::types::DataType>) -> Self {
        use crate::records::types::ColumnDef;
        let column_defs: Vec<ColumnDef> = column_types
            .iter()
            .enumerate()
            .map(|(i, dt)| ColumnDef::new(format!("col{}", i), *dt))
            .collect();
        let cached_schema = crate::records::Schema::new(column_defs);
        let decode_columns: Vec<usize> = (0..column_types.len()).collect();
        Self {
            column_types,
            decode_columns,
            cached_schema,
        }
    }

    pub fn with_projections(
        column_types: Vec<crate::records::types::DataType>,
        projections: Vec<usize>,
    ) -> Self {
        use crate::records::types::ColumnDef;
        let column_defs: Vec<ColumnDef> = column_types
            .iter()
            .enumerate()
            .map(|(i, dt)| ColumnDef::new(format!("col{}", i), *dt))
            .collect();
        let cached_schema = crate::records::Schema::new(column_defs);
        Self {
            column_types,
            decode_columns: projections,
            cached_schema,
        }
    }
}

impl RecordDecoder for SimpleDecoder {
    fn decode(&self, _key: &[u8], value: &[u8]) -> eyre::Result<Vec<Value<'static>>> {
        self.decode_columns_vec(_key, value, &self.decode_columns)
    }
}

impl SimpleDecoder {
    fn decode_columns_vec(
        &self,
        _key: &[u8],
        value: &[u8],
        columns: &[usize],
    ) -> eyre::Result<Vec<Value<'static>>> {
        let mut output = Vec::with_capacity(columns.len());
        self.decode_columns_into(_key, value, columns, &mut output)?;
        Ok(output)
    }

    fn decode_columns_into(
        &self,
        _key: &[u8],
        value: &[u8],
        columns: &[usize],
        output: &mut Vec<Value<'static>>,
    ) -> eyre::Result<()> {
        use crate::records::types::DataType;
        use crate::records::RecordView;

        if value.is_empty() {
            output.extend(std::iter::repeat_n(Value::Null, columns.len()));
            return Ok(());
        }

        let view = RecordView::new(value, &self.cached_schema)?;

        for &idx in columns {
            if idx >= self.column_types.len() {
                output.push(Value::Null);
                continue;
            }
            if view.is_null(idx) {
                output.push(Value::Null);
                continue;
            }
            let dt = &self.column_types[idx];
            let val = match dt {
                DataType::Int2 => Value::Int(view.get_int2(idx)? as i64),
                DataType::Int4 => Value::Int(view.get_int4(idx)? as i64),
                DataType::Int8 => Value::Int(view.get_int8(idx)?),
                DataType::Float4 => Value::Float(view.get_float4(idx)? as f64),
                DataType::Float8 => Value::Float(view.get_float8(idx)?),
                DataType::Bool => Value::Int(if view.get_bool(idx)? { 1 } else { 0 }),
                DataType::Text | DataType::Varchar | DataType::Char => {
                    Value::Text(Cow::Owned(view.get_text(idx)?.to_string()))
                }
                DataType::Blob => Value::Blob(Cow::Owned(view.get_blob(idx)?.to_vec())),
                DataType::Uuid => Value::Uuid(*view.get_uuid(idx)?),
                DataType::Vector => Value::Vector(Cow::Owned(view.get_vector(idx)?.to_vec())),
                DataType::Jsonb => {
                    let jsonb_view = view.get_jsonb(idx)?;
                    Value::Jsonb(Cow::Owned(jsonb_view.data().to_vec()))
                }
                DataType::Date => Value::Int(view.get_date(idx)? as i64),
                DataType::Time => Value::Int(view.get_time(idx)?),
                DataType::Timestamp => Value::Int(view.get_timestamp(idx)?),
                DataType::MacAddr => Value::MacAddr(*view.get_macaddr(idx)?),
                DataType::Inet4 => Value::Inet4(*view.get_inet4(idx)?),
                DataType::Inet6 => Value::Inet6(*view.get_inet6(idx)?),
                DataType::TimestampTz => {
                    let (micros, offset_secs) = view.get_timestamptz(idx)?;
                    Value::TimestampTz {
                        micros,
                        offset_secs,
                    }
                }
                DataType::Interval => {
                    let (micros, days, months) = view.get_interval(idx)?;
                    Value::Interval {
                        micros,
                        days,
                        months,
                    }
                }
                DataType::Point => {
                    let (x, y) = view.get_point(idx)?;
                    Value::Point { x, y }
                }
                DataType::Box => {
                    let (low, high) = view.get_box(idx)?;
                    Value::GeoBox { low, high }
                }
                DataType::Circle => {
                    let (center, radius) = view.get_circle(idx)?;
                    Value::Circle { center, radius }
                }
                DataType::Enum => {
                    let (type_id, ordinal) = view.get_enum(idx)?;
                    Value::Enum { type_id, ordinal }
                }
                DataType::Decimal => {
                    let dec = view.get_decimal(idx)?;
                    Value::Decimal {
                        digits: dec.digits(),
                        scale: dec.scale(),
                    }
                }
                DataType::Array => {
                    let arr = view.get_array(idx)?;
                    Value::Text(Cow::Owned(Self::format_array(&arr)?))
                }
                DataType::Composite => {
                    let comp = view.get_composite(idx, 0)?;
                    Value::Text(Cow::Owned(Self::format_composite(&comp)))
                }
                DataType::Int4Range => {
                    let range = view.get_int4_range(idx)?;
                    Value::Text(Cow::Owned(Self::format_int_range(
                        range.lower.map(|v| v as i64),
                        range.upper.map(|v| v as i64),
                        range.lower_inclusive,
                        range.upper_inclusive,
                        range.is_empty,
                    )))
                }
                DataType::Int8Range => {
                    let range = view.get_int8_range(idx)?;
                    Value::Text(Cow::Owned(Self::format_int_range(
                        range.lower,
                        range.upper,
                        range.lower_inclusive,
                        range.upper_inclusive,
                        range.is_empty,
                    )))
                }
                DataType::DateRange => {
                    let range = view.get_date_range(idx)?;
                    Value::Text(Cow::Owned(Self::format_int_range(
                        range.lower.map(|v| v as i64),
                        range.upper.map(|v| v as i64),
                        range.lower_inclusive,
                        range.upper_inclusive,
                        range.is_empty,
                    )))
                }
                DataType::TimestampRange => {
                    let range = view.get_timestamp_range(idx)?;
                    Value::Text(Cow::Owned(Self::format_int_range(
                        range.lower,
                        range.upper,
                        range.lower_inclusive,
                        range.upper_inclusive,
                        range.is_empty,
                    )))
                }
            };
            output.push(val);
        }
        Ok(())
    }

    fn format_array(arr: &crate::records::array::ArrayView<'_>) -> eyre::Result<String> {
        use crate::records::types::DataType;

        if arr.is_empty() {
            return Ok("{}".to_string());
        }

        let mut parts = Vec::with_capacity(arr.len());
        let elem_type = arr.elem_type();

        for i in 0..arr.len() {
            if arr.is_null(i) {
                parts.push("NULL".to_string());
                continue;
            }
            let elem_str = match elem_type {
                DataType::Int2 => arr.get_int2(i)?.to_string(),
                DataType::Int4 => arr.get_int4(i)?.to_string(),
                DataType::Int8 => arr.get_int8(i)?.to_string(),
                DataType::Float4 => arr.get_float4(i)?.to_string(),
                DataType::Float8 => arr.get_float8(i)?.to_string(),
                DataType::Bool => {
                    if arr.get_bool(i)? {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                }
                DataType::Text | DataType::Varchar | DataType::Char => {
                    let s = arr.get_text(i)?;
                    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                }
                DataType::Blob => {
                    let b = arr.get_blob(i)?;
                    format!(
                        "\\x{}",
                        b.iter()
                            .map(|byte| format!("{:02x}", byte))
                            .collect::<String>()
                    )
                }
                _ => "?".to_string(),
            };
            parts.push(elem_str);
        }

        Ok(format!("{{{}}}", parts.join(",")))
    }

    fn format_composite(comp: &crate::records::composite::CompositeView<'_>) -> String {
        let mut parts = Vec::with_capacity(comp.field_count());
        for i in 0..comp.field_count() {
            if comp.is_null(i) {
                parts.push("".to_string());
            } else if let Ok(field_data) = comp.get_field(i) {
                parts.push(format!("<{} bytes>", field_data.len()));
            } else {
                parts.push("".to_string());
            }
        }
        format!("({})", parts.join(","))
    }

    fn format_int_range(
        lower: Option<i64>,
        upper: Option<i64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
        is_empty: bool,
    ) -> String {
        if is_empty {
            return "empty".to_string();
        }
        let left_bracket = if lower_inclusive { '[' } else { '(' };
        let right_bracket = if upper_inclusive { ']' } else { ')' };
        let lower_str = lower.map(|v| v.to_string()).unwrap_or_default();
        let upper_str = upper.map(|v| v.to_string()).unwrap_or_default();
        format!(
            "{}{},{}{}",
            left_bracket, lower_str, upper_str, right_bracket
        )
    }

    pub fn decode_into(
        &self,
        _key: &[u8],
        value: &[u8],
        output: &mut Vec<Value<'static>>,
    ) -> eyre::Result<()> {
        self.decode_columns_into(_key, value, &self.decode_columns, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::types::DataType;

    #[test]
    fn test_format_int_range_inclusive() {
        let result = SimpleDecoder::format_int_range(Some(1), Some(10), true, true, false);
        assert_eq!(result, "[1,10]");
    }

    #[test]
    fn test_format_int_range_exclusive() {
        let result = SimpleDecoder::format_int_range(Some(1), Some(10), false, false, false);
        assert_eq!(result, "(1,10)");
    }

    #[test]
    fn test_format_int_range_mixed() {
        let result = SimpleDecoder::format_int_range(Some(1), Some(10), true, false, false);
        assert_eq!(result, "[1,10)");
    }

    #[test]
    fn test_format_int_range_empty() {
        let result = SimpleDecoder::format_int_range(None, None, false, false, true);
        assert_eq!(result, "empty");
    }

    #[test]
    fn test_format_int_range_unbounded_lower() {
        let result = SimpleDecoder::format_int_range(None, Some(10), false, false, false);
        assert_eq!(result, "(,10)");
    }

    #[test]
    fn test_format_int_range_unbounded_upper() {
        let result = SimpleDecoder::format_int_range(Some(1), None, true, false, false);
        assert_eq!(result, "[1,)");
    }

    #[test]
    fn test_format_array_int4() {
        use crate::records::array::ArrayBuilder;

        let mut builder = ArrayBuilder::new(DataType::Int4);
        builder.push_int4(1);
        builder.push_int4(2);
        builder.push_int4(3);
        let data = builder.build();

        let view = crate::records::array::ArrayView::new(&data).unwrap();
        let result = SimpleDecoder::format_array(&view).unwrap();
        assert_eq!(result, "{1,2,3}");
    }

    #[test]
    fn test_format_array_with_null() {
        use crate::records::array::ArrayBuilder;

        let mut builder = ArrayBuilder::new(DataType::Int4);
        builder.push_int4(1);
        builder.push_null();
        builder.push_int4(3);
        let data = builder.build();

        let view = crate::records::array::ArrayView::new(&data).unwrap();
        let result = SimpleDecoder::format_array(&view).unwrap();
        assert_eq!(result, "{1,NULL,3}");
    }

    #[test]
    fn test_format_array_text() {
        use crate::records::array::ArrayBuilder;

        let mut builder = ArrayBuilder::new(DataType::Text);
        builder.push_text("hello");
        builder.push_text("world");
        let data = builder.build();

        let view = crate::records::array::ArrayView::new(&data).unwrap();
        let result = SimpleDecoder::format_array(&view).unwrap();
        assert_eq!(result, "{\"hello\",\"world\"}");
    }

    #[test]
    fn test_format_array_empty() {
        use crate::records::array::ArrayBuilder;

        let builder = ArrayBuilder::new(DataType::Int4);
        let data = builder.build();

        let view = crate::records::array::ArrayView::new(&data).unwrap();
        let result = SimpleDecoder::format_array(&view).unwrap();
        assert_eq!(result, "{}");
    }

    #[test]
    fn test_format_composite() {
        use crate::records::composite::CompositeView;

        let data = vec![0x04, 0x00, 0x00];
        let view = CompositeView::new(&data, 0).unwrap();
        let result = SimpleDecoder::format_composite(&view);
        assert_eq!(result, "()");
    }

    #[test]
    fn test_decode_array_column() {
        use crate::records::array::ArrayBuilder;
        use crate::records::types::ColumnDef;
        use crate::records::{RecordBuilder, Schema};

        let column_defs = vec![ColumnDef::new("arr", DataType::Array)];
        let schema = Schema::new(column_defs);
        let mut builder = RecordBuilder::new(&schema);

        let mut arr_builder = ArrayBuilder::new(DataType::Int4);
        arr_builder.push_int4(10);
        arr_builder.push_int4(20);
        arr_builder.push_int4(30);
        let arr_data = arr_builder.build();

        builder.set_blob(0, &arr_data).unwrap();
        let record_data = builder.build().unwrap();

        let decoder = SimpleDecoder::new(vec![DataType::Array]);
        let result = decoder.decode(&[], &record_data).unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            Value::Text(s) => assert_eq!(s.as_ref(), "{10,20,30}"),
            _ => panic!("Expected Text value, got {:?}", result[0]),
        }
    }

    #[test]
    fn test_decode_int4_range_column() {
        use crate::records::types::ColumnDef;
        use crate::records::{RecordBuilder, Schema};

        let column_defs = vec![ColumnDef::new("r", DataType::Int4Range)];
        let schema = Schema::new(column_defs);
        let mut builder = RecordBuilder::new(&schema);

        builder
            .set_int4_range(0, Some(1), Some(10), true, false)
            .unwrap();
        let record_data = builder.build().unwrap();

        let decoder = SimpleDecoder::new(vec![DataType::Int4Range]);
        let result = decoder.decode(&[], &record_data).unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            Value::Text(s) => assert_eq!(s.as_ref(), "[1,10)"),
            _ => panic!("Expected Text value, got {:?}", result[0]),
        }
    }

    #[test]
    fn test_decode_int8_range_column() {
        use crate::records::types::ColumnDef;
        use crate::records::{RecordBuilder, Schema};

        let column_defs = vec![ColumnDef::new("r", DataType::Int8Range)];
        let schema = Schema::new(column_defs);
        let mut builder = RecordBuilder::new(&schema);

        builder
            .set_int8_range(0, Some(100), Some(200), true, true)
            .unwrap();
        let record_data = builder.build().unwrap();

        let decoder = SimpleDecoder::new(vec![DataType::Int8Range]);
        let result = decoder.decode(&[], &record_data).unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            Value::Text(s) => assert_eq!(s.as_ref(), "[100,200]"),
            _ => panic!("Expected Text value, got {:?}", result[0]),
        }
    }

    #[test]
    fn test_decode_empty_range() {
        use crate::records::types::ColumnDef;
        use crate::records::{RecordBuilder, Schema};

        let column_defs = vec![ColumnDef::new("r", DataType::Int4Range)];
        let schema = Schema::new(column_defs);
        let mut builder = RecordBuilder::new(&schema);

        builder.set_int4_range_empty(0).unwrap();
        let record_data = builder.build().unwrap();

        let decoder = SimpleDecoder::new(vec![DataType::Int4Range]);
        let result = decoder.decode(&[], &record_data).unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            Value::Text(s) => assert_eq!(s.as_ref(), "empty"),
            _ => panic!("Expected Text value, got {:?}", result[0]),
        }
    }
}
