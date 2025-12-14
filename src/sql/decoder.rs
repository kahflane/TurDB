use std::borrow::Cow;
use crate::types::Value;

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
                DataType::Text => Value::Text(Cow::Owned(view.get_text(idx)?.to_string())),
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
                _ => Value::Null,
            };
            output.push(val);
        }
        Ok(())
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