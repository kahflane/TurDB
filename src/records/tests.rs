//! Tests for the records module

use super::*;

#[test]
fn record_view_can_be_created_with_data_and_schema() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
    ]);

    let data = vec![0x00, 0x00];

    let view = RecordView::new(&data, &schema);
    assert!(view.is_ok());

    let view = view.unwrap();
    assert_eq!(view.data().len(), 2);
    assert_eq!(view.schema().column_count(), 2);
}

#[test]
fn record_view_rejects_empty_data() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
    let data: Vec<u8> = vec![];

    let result = RecordView::new(&data, &schema);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
fn record_view_borrows_data_zero_copy() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
    let data = vec![0x10, 0x00, 0x01, 0x02, 0x03, 0x04];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(std::ptr::eq(view.data().as_ptr(), data.as_ptr()));
}

#[test]
fn schema_tracks_fixed_and_variable_columns() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
        ColumnDef::new("age", DataType::Int2),
        ColumnDef::new("bio", DataType::Blob),
    ]);

    assert_eq!(schema.column_count(), 4);
    assert_eq!(schema.var_column_count(), 2);

    assert_eq!(schema.var_column_index(1), Some(0));
    assert_eq!(schema.var_column_index(3), Some(1));
    assert_eq!(schema.var_column_index(0), None);
    assert_eq!(schema.var_column_index(2), None);
}

#[test]
fn schema_calculates_fixed_offsets() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int8),
        ColumnDef::new("c", DataType::Text),
        ColumnDef::new("d", DataType::Int2),
    ]);

    assert_eq!(schema.fixed_offset(0), 0);
    assert_eq!(schema.fixed_offset(1), 4);
    assert_eq!(schema.fixed_offset(2), 12);
    assert_eq!(schema.fixed_offset(3), 12);

    assert_eq!(schema.total_fixed_size(), 14);
}

#[test]
fn data_type_fixed_sizes() {
    assert_eq!(DataType::Bool.fixed_size(), Some(1));
    assert_eq!(DataType::Int2.fixed_size(), Some(2));
    assert_eq!(DataType::Int4.fixed_size(), Some(4));
    assert_eq!(DataType::Int8.fixed_size(), Some(8));
    assert_eq!(DataType::Float4.fixed_size(), Some(4));
    assert_eq!(DataType::Float8.fixed_size(), Some(8));
    assert_eq!(DataType::Date.fixed_size(), Some(4));
    assert_eq!(DataType::Time.fixed_size(), Some(8));
    assert_eq!(DataType::Timestamp.fixed_size(), Some(8));
    assert_eq!(DataType::Uuid.fixed_size(), Some(16));
    assert_eq!(DataType::MacAddr.fixed_size(), Some(6));
    assert_eq!(DataType::Text.fixed_size(), None);
    assert_eq!(DataType::Blob.fixed_size(), None);
}

#[test]
fn data_type_is_variable() {
    assert!(!DataType::Int4.is_variable());
    assert!(DataType::Text.is_variable());
    assert!(DataType::Blob.is_variable());
}

#[test]
fn record_view_header_len_parses_little_endian() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
    let data = vec![0x05, 0x00, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.header_len(), 5);
}

#[test]
fn record_view_header_len_larger_value() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
    let mut data = vec![0x00, 0x01];
    data.resize(256 + 10, 0);

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.header_len(), 256);
}

#[test]
fn record_view_null_bitmap_size_calculation() {
    assert_eq!(Schema::null_bitmap_size(1), 1);
    assert_eq!(Schema::null_bitmap_size(8), 1);
    assert_eq!(Schema::null_bitmap_size(9), 2);
    assert_eq!(Schema::null_bitmap_size(16), 2);
    assert_eq!(Schema::null_bitmap_size(17), 3);
}

#[test]
fn record_view_null_bitmap_slice() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
        ColumnDef::new("c", DataType::Int4),
    ]);

    let data = vec![0x05, 0x00, 0b0000_0101, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();
    let bitmap = view.null_bitmap();
    assert_eq!(bitmap.len(), 1);
    assert_eq!(bitmap[0], 0b0000_0101);
}

#[test]
fn record_view_offset_table_slice() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
        ColumnDef::new("bio", DataType::Blob),
    ]);

    let data = vec![0x09, 0x00, 0x00, 0x10, 0x00, 0x20, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();
    let offsets = view.offset_table();
    assert_eq!(offsets.len(), 4);
    assert_eq!(offsets[0], 0x10);
    assert_eq!(offsets[1], 0x00);
    assert_eq!(offsets[2], 0x20);
    assert_eq!(offsets[3], 0x00);
}

#[test]
fn record_view_data_payload_offset() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
    let data = vec![0x04, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.data_offset(), 4);
}

#[test]
fn is_null_checks_bitmap_bit_correctly() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
        ColumnDef::new("c", DataType::Int4),
        ColumnDef::new("d", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0b0000_0101];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(view.is_null(0));
    assert!(!view.is_null(1));
    assert!(view.is_null(2));
    assert!(!view.is_null(3));
}

#[test]
fn is_null_handles_multi_byte_bitmap() {
    let schema = Schema::new(vec![
        ColumnDef::new("c0", DataType::Int4),
        ColumnDef::new("c1", DataType::Int4),
        ColumnDef::new("c2", DataType::Int4),
        ColumnDef::new("c3", DataType::Int4),
        ColumnDef::new("c4", DataType::Int4),
        ColumnDef::new("c5", DataType::Int4),
        ColumnDef::new("c6", DataType::Int4),
        ColumnDef::new("c7", DataType::Int4),
        ColumnDef::new("c8", DataType::Int4),
        ColumnDef::new("c9", DataType::Int4),
    ]);

    let data = vec![0x04, 0x00, 0b1000_0001, 0b0000_0010];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(view.is_null(0));
    assert!(!view.is_null(1));
    assert!(!view.is_null(2));
    assert!(!view.is_null(3));
    assert!(!view.is_null(4));
    assert!(!view.is_null(5));
    assert!(!view.is_null(6));
    assert!(view.is_null(7));
    assert!(!view.is_null(8));
    assert!(view.is_null(9));
}

#[test]
fn is_null_all_null_columns() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0b0000_0011];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(view.is_null(0));
    assert!(view.is_null(1));
}

#[test]
fn is_null_no_null_columns() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0b0000_0000];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(!view.is_null(0));
    assert!(!view.is_null(1));
}

#[test]
fn get_fixed_col_offset_calculates_correctly() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int8),
        ColumnDef::new("c", DataType::Int2),
    ]);

    let data = vec![0x03, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_fixed_col_offset(0), 3);
    assert_eq!(view.get_fixed_col_offset(1), 7);
    assert_eq!(view.get_fixed_col_offset(2), 15);
}

#[test]
fn get_var_bounds_reads_offset_table() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
        ColumnDef::new("bio", DataType::Blob),
    ]);

    let data = vec![
        0x0B, 0x00, 0x00, 0x05, 0x00, 0x0B, 0x00, 0x01, 0x02, 0x03, 0x04, b'h', b'e', b'l', b'l',
        b'o', b'b', b'i', b'o', b'!', b'!', b'!',
    ];

    let view = RecordView::new(&data, &schema).unwrap();

    let (start, end) = view.get_var_bounds(1).unwrap();
    assert_eq!(start, 15);
    assert_eq!(end, 20);

    let (start, end) = view.get_var_bounds(2).unwrap();
    assert_eq!(start, 20);
    assert_eq!(end, 26);
}

#[test]
fn get_int4_reads_little_endian() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);

    let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int4(0).unwrap(), 42);
}

#[test]
fn get_int4_negative_value() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend((-100i32).to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int4(0).unwrap(), -100);
}

#[test]
fn get_int2_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("val", DataType::Int2)]);

    let data = vec![0x03, 0x00, 0x00, 0xD2, 0x04];

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int2(0).unwrap(), 1234);
}

#[test]
fn get_int8_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("val", DataType::Int8)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(123456789012345_i64.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int8(0).unwrap(), 123456789012345);
}

#[test]
fn get_float4_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("val", DataType::Float4)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(1.25_f32.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    let val = view.get_float4(0).unwrap();
    assert!((val - 1.25).abs() < 0.001);
}

#[test]
fn get_float8_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("val", DataType::Float8)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(1.23456789012345_f64.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    let val = view.get_float8(0).unwrap();
    assert!((val - 1.23456789012345).abs() < 1e-10);
}

#[test]
fn get_text_returns_zero_copy_str() {
    let schema = Schema::new(vec![ColumnDef::new("name", DataType::Text)]);

    let data = vec![0x05, 0x00, 0x00, 0x05, 0x00, b'h', b'e', b'l', b'l', b'o'];

    let view = RecordView::new(&data, &schema).unwrap();

    let text = view.get_text(0).unwrap();
    assert_eq!(text, "hello");

    let text_ptr = text.as_ptr();
    let data_ptr = data.as_ptr();
    assert!(text_ptr >= data_ptr && text_ptr < unsafe { data_ptr.add(data.len()) });
}

#[test]
fn get_blob_returns_zero_copy_bytes() {
    let schema = Schema::new(vec![ColumnDef::new("data", DataType::Blob)]);

    let data = vec![0x05, 0x00, 0x00, 0x03, 0x00, 0xDE, 0xAD, 0xBE];

    let view = RecordView::new(&data, &schema).unwrap();

    let blob = view.get_blob(0).unwrap();
    assert_eq!(blob, &[0xDE, 0xAD, 0xBE]);

    let blob_ptr = blob.as_ptr();
    let data_ptr = data.as_ptr();
    assert!(blob_ptr >= data_ptr && blob_ptr < unsafe { data_ptr.add(data.len()) });
}

#[test]
fn get_date_reads_days_since_epoch() {
    let schema = Schema::new(vec![ColumnDef::new("d", DataType::Date)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(19000_i32.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_date(0).unwrap(), 19000);
}

#[test]
fn get_time_reads_microseconds() {
    let schema = Schema::new(vec![ColumnDef::new("t", DataType::Time)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(43200000000_i64.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_time(0).unwrap(), 43200000000);
}

#[test]
fn get_timestamp_reads_microseconds_since_epoch() {
    let schema = Schema::new(vec![ColumnDef::new("ts", DataType::Timestamp)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(1702300000000000_i64.to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_timestamp(0).unwrap(), 1702300000000000);
}

#[test]
fn get_uuid_returns_reference() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Uuid)]);

    let uuid_bytes: [u8; 16] = [
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88,
    ];

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(uuid_bytes);

    let view = RecordView::new(&data, &schema).unwrap();

    let uuid = view.get_uuid(0).unwrap();
    assert_eq!(uuid, &uuid_bytes);

    let uuid_ptr = uuid.as_ptr();
    let data_ptr = data.as_ptr();
    assert!(uuid_ptr >= data_ptr && uuid_ptr < unsafe { data_ptr.add(data.len()) });
}

#[test]
fn get_bool_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("flag", DataType::Bool)]);

    let data_true = vec![0x03, 0x00, 0x00, 0x01];

    let data_false = vec![0x03, 0x00, 0x00, 0x00];

    let view_true = RecordView::new(&data_true, &schema).unwrap();
    let view_false = RecordView::new(&data_false, &schema).unwrap();

    assert!(view_true.get_bool(0).unwrap());
    assert!(!view_false.get_bool(0).unwrap());
}

#[test]
fn get_macaddr_returns_reference() {
    let schema = Schema::new(vec![ColumnDef::new("mac", DataType::MacAddr)]);

    let mac_bytes: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(mac_bytes);

    let view = RecordView::new(&data, &schema).unwrap();

    let mac = view.get_macaddr(0).unwrap();
    assert_eq!(mac, &mac_bytes);
}

#[test]
fn schema_evolution_record_column_count() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0b0000_0000, 0x01, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.record_column_count(), 1);
}

#[test]
fn schema_evolution_column_beyond_record_is_null() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("age", DataType::Int4),
        ColumnDef::new("score", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(!view.is_null_or_missing(0));
    assert!(view.is_null_or_missing(1));
    assert!(view.is_null_or_missing(2));
}

#[test]
fn schema_evolution_get_optional_returns_none_for_missing() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("age", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int4_opt(0).unwrap(), Some(42));
    assert_eq!(view.get_int4_opt(1).unwrap(), None);
}

#[test]
fn schema_evolution_mixed_null_and_missing() {
    let schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
        ColumnDef::new("c", DataType::Int4),
    ]);

    let data = vec![0x03, 0x00, 0b0000_0001, 0x2A, 0x00, 0x00, 0x00];

    let view = RecordView::new(&data, &schema).unwrap();

    assert!(view.is_null_or_missing(0));
    assert!(view.is_null_or_missing(2));

    assert_eq!(view.get_int4_opt(0).unwrap(), None);
    assert_eq!(view.get_int4_opt(2).unwrap(), None);
}

#[test]
fn get_bool_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("flag", DataType::Bool)]);
    let data = vec![0x03, 0x00, 0b0000_0001, 0x01];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_bool_opt(0).unwrap(), None);
}

#[test]
fn get_date_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("d", DataType::Date)]);
    let data = vec![0x03, 0x00, 0b0000_0001];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_date_opt(0).unwrap(), None);
}

#[test]
fn get_time_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("t", DataType::Time)]);
    let data = vec![0x03, 0x00, 0b0000_0001];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_time_opt(0).unwrap(), None);
}

#[test]
fn get_timestamp_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("ts", DataType::Timestamp)]);
    let data = vec![0x03, 0x00, 0b0000_0001];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_timestamp_opt(0).unwrap(), None);
}

#[test]
fn get_uuid_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("id", DataType::Uuid)]);
    let data = vec![0x03, 0x00, 0b0000_0001];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_uuid_opt(0).unwrap(), None);
}

#[test]
fn get_macaddr_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("mac", DataType::MacAddr)]);
    let data = vec![0x03, 0x00, 0b0000_0001];

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_macaddr_opt(0).unwrap(), None);
}

#[test]
fn data_type_timestamptz_fixed_size() {
    assert_eq!(DataType::TimestampTz.fixed_size(), Some(12));
}

#[test]
fn data_type_inet4_fixed_size() {
    assert_eq!(DataType::Inet4.fixed_size(), Some(4));
}

#[test]
fn data_type_inet6_fixed_size() {
    assert_eq!(DataType::Inet6.fixed_size(), Some(16));
}

#[test]
fn data_type_vector_is_variable() {
    assert!(DataType::Vector.is_variable());
}

#[test]
fn get_timestamptz_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("ts", DataType::TimestampTz)]);

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(1702300000000000_i64.to_le_bytes());
    data.extend((-300_i32).to_le_bytes());

    let view = RecordView::new(&data, &schema).unwrap();

    let (micros, offset_secs) = view.get_timestamptz(0).unwrap();
    assert_eq!(micros, 1702300000000000);
    assert_eq!(offset_secs, -300);
}

#[test]
fn get_inet4_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet4)]);

    let data = vec![0x03, 0x00, 0x00, 192, 168, 1, 1];

    let view = RecordView::new(&data, &schema).unwrap();

    let ip = view.get_inet4(0).unwrap();
    assert_eq!(ip, &[192, 168, 1, 1]);
}

#[test]
fn get_inet6_reads_correctly() {
    let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet6)]);

    let ipv6: [u8; 16] = [
        0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70, 0x73,
        0x34,
    ];

    let mut data = vec![0x03, 0x00, 0x00];
    data.extend(ipv6);

    let view = RecordView::new(&data, &schema).unwrap();

    let ip = view.get_inet6(0).unwrap();
    assert_eq!(ip, &ipv6);
}

#[test]
fn record_builder_creates_simple_record() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("age", DataType::Int2),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 42).unwrap();
    builder.set_int2(1, 25).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_int4(0).unwrap(), 42);
    assert_eq!(view.get_int2(1).unwrap(), 25);
}

#[test]
fn record_builder_handles_null_values() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("age", DataType::Int4),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 42).unwrap();
    builder.set_null(1);

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_int4(0).unwrap(), 42);
    assert!(view.is_null(1));
}

#[test]
fn record_builder_with_variable_columns() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 1).unwrap();
    builder.set_text(1, "hello").unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_int4(0).unwrap(), 1);
    assert_eq!(view.get_text(1).unwrap(), "hello");
}

#[test]
fn record_builder_with_multiple_variable_columns() {
    let schema = Schema::new(vec![
        ColumnDef::new("name", DataType::Text),
        ColumnDef::new("bio", DataType::Blob),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_text(0, "alice").unwrap();
    builder.set_blob(1, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_text(0).unwrap(), "alice");
    assert_eq!(view.get_blob(1).unwrap(), &[0xDE, 0xAD, 0xBE, 0xEF]);
}

#[test]
fn record_builder_roundtrip_all_fixed_types() {
    let schema = Schema::new(vec![
        ColumnDef::new("b", DataType::Bool),
        ColumnDef::new("i2", DataType::Int2),
        ColumnDef::new("i4", DataType::Int4),
        ColumnDef::new("i8", DataType::Int8),
        ColumnDef::new("f4", DataType::Float4),
        ColumnDef::new("f8", DataType::Float8),
        ColumnDef::new("d", DataType::Date),
        ColumnDef::new("t", DataType::Time),
        ColumnDef::new("ts", DataType::Timestamp),
        ColumnDef::new("u", DataType::Uuid),
        ColumnDef::new("mac", DataType::MacAddr),
    ]);

    let uuid: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mac: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

    let mut builder = RecordBuilder::new(&schema);
    builder.set_bool(0, true).unwrap();
    builder.set_int2(1, 1234).unwrap();
    builder.set_int4(2, 567890).unwrap();
    builder.set_int8(3, 123456789012345).unwrap();
    builder.set_float4(4, 1.5).unwrap();
    builder.set_float8(5, 2.5).unwrap();
    builder.set_date(6, 19000).unwrap();
    builder.set_time(7, 43200000000).unwrap();
    builder.set_timestamp(8, 1702300000000000).unwrap();
    builder.set_uuid(9, &uuid).unwrap();
    builder.set_macaddr(10, &mac).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert!(view.get_bool(0).unwrap());
    assert_eq!(view.get_int2(1).unwrap(), 1234);
    assert_eq!(view.get_int4(2).unwrap(), 567890);
    assert_eq!(view.get_int8(3).unwrap(), 123456789012345);
    assert!((view.get_float4(4).unwrap() - 1.5).abs() < 0.001);
    assert!((view.get_float8(5).unwrap() - 2.5).abs() < 0.001);
    assert_eq!(view.get_date(6).unwrap(), 19000);
    assert_eq!(view.get_time(7).unwrap(), 43200000000);
    assert_eq!(view.get_timestamp(8).unwrap(), 1702300000000000);
    assert_eq!(view.get_uuid(9).unwrap(), &uuid);
    assert_eq!(view.get_macaddr(10).unwrap(), &mac);
}

#[test]
fn record_builder_reset_allows_reuse() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("name", DataType::Text),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 100).unwrap();
    builder.set_text(1, "first").unwrap();
    let data1 = builder.build().unwrap();

    let view1 = RecordView::new(&data1, &schema).unwrap();
    assert_eq!(view1.get_int4(0).unwrap(), 100);
    assert_eq!(view1.get_text(1).unwrap(), "first");

    builder.reset();
    builder.set_int4(0, 200).unwrap();
    builder.set_text(1, "second").unwrap();
    let data2 = builder.build().unwrap();

    let view2 = RecordView::new(&data2, &schema).unwrap();
    assert_eq!(view2.get_int4(0).unwrap(), 200);
    assert_eq!(view2.get_text(1).unwrap(), "second");
}

#[test]
fn record_builder_set_timestamptz_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("ts", DataType::TimestampTz)]);

    let mut builder = RecordBuilder::new(&schema);
    builder
        .set_timestamptz(0, 1702300000000000, -18000)
        .unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    let (micros, offset_secs) = view.get_timestamptz(0).unwrap();
    assert_eq!(micros, 1702300000000000);
    assert_eq!(offset_secs, -18000);
}

#[test]
fn record_builder_set_inet4_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet4)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_inet4(0, &[192, 168, 1, 1]).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_inet4(0).unwrap(), &[192, 168, 1, 1]);
}

#[test]
fn record_builder_set_inet6_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet6)]);

    let ipv6: [u8; 16] = [
        0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70, 0x73,
        0x34,
    ];

    let mut builder = RecordBuilder::new(&schema);
    builder.set_inet6(0, &ipv6).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_inet6(0).unwrap(), &ipv6);
}

#[test]
fn record_builder_set_vector_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

    let vector = vec![1.0_f32, 2.5, -3.0, 0.0, 4.25];

    let mut builder = RecordBuilder::new(&schema);
    builder.set_vector(0, &vector).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    let result = view.get_vector_copy(0).unwrap();
    assert_eq!(result.len(), 5);
    for (a, b) in result.iter().zip(vector.iter()) {
        assert!((a - b).abs() < 0.0001);
    }
}

#[test]
fn record_builder_vector_with_fixed_column() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("embedding", DataType::Vector),
    ]);

    let vector = vec![0.5_f32, 1.5, 2.5];

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 42).unwrap();
    builder.set_vector(1, &vector).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_int4(0).unwrap(), 42);
    let result = view.get_vector_copy(1).unwrap();
    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.5).abs() < 0.0001);
    assert!((result[1] - 1.5).abs() < 0.0001);
    assert!((result[2] - 2.5).abs() < 0.0001);
}

#[test]
fn get_vector_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_null(0);

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    assert_eq!(view.get_vector_opt(0).unwrap(), None);
}

#[test]
fn record_builder_empty_vector() {
    let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

    let vector: Vec<f32> = vec![];

    let mut builder = RecordBuilder::new(&schema);
    builder.set_vector(0, &vector).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    let result = view.get_vector_copy(0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn record_builder_large_vector() {
    let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

    let vector: Vec<f32> = (0..1024).map(|i| i as f32 / 100.0).collect();

    let mut builder = RecordBuilder::new(&schema);
    builder.set_vector(0, &vector).unwrap();

    let data = builder.build().unwrap();

    let view = RecordView::new(&data, &schema).unwrap();
    let result = view.get_vector_copy(0).unwrap();
    assert_eq!(result.len(), 1024);
    for (i, (&a, &b)) in result.iter().zip(vector.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.0001,
            "mismatch at index {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn data_type_jsonb_is_variable() {
    assert!(DataType::Jsonb.is_variable());
    assert_eq!(DataType::Jsonb.fixed_size(), None);
}

#[test]
fn jsonb_builder_simple_object() {
    let mut builder = JsonbBuilder::new_object();
    builder.set("name", "Alice");
    builder.set("age", 30);

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(view.root_type(), jsonb::JSONB_TYPE_OBJECT);
    assert_eq!(view.object_len().unwrap(), 2);
}

#[test]
fn jsonb_builder_object_get_key() {
    let mut builder = JsonbBuilder::new_object();
    builder.set("name", "Bob");
    builder.set("score", 95.5);
    builder.set("active", true);

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    let name = view.get("name").unwrap().unwrap();
    assert_eq!(name, JsonbValue::String("Bob"));

    let score = view.get("score").unwrap().unwrap();
    if let JsonbValue::Number(n) = score {
        assert!((n - 95.5).abs() < 0.0001);
    } else {
        panic!("expected number");
    }

    let active = view.get("active").unwrap().unwrap();
    assert_eq!(active, JsonbValue::Bool(true));

    assert!(view.get("missing").unwrap().is_none());
}

#[test]
fn jsonb_builder_array() {
    let mut builder = JsonbBuilder::new_array();
    builder.push(1);
    builder.push(2);
    builder.push(3);

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(view.root_type(), jsonb::JSONB_TYPE_ARRAY);
    assert_eq!(view.array_len().unwrap(), 3);

    let first = view.array_get(0).unwrap().unwrap();
    if let JsonbValue::Number(n) = first {
        assert!((n - 1.0).abs() < 0.0001);
    } else {
        panic!("expected number");
    }
}

#[test]
fn jsonb_builder_nested_object() {
    let mut outer = JsonbBuilder::new_object();
    outer.set("name", "Charlie");
    outer.set(
        "address",
        JsonbBuilderValue::Object(vec![
            (
                "city".to_string(),
                JsonbBuilderValue::String("NYC".to_string()),
            ),
            (
                "zip".to_string(),
                JsonbBuilderValue::String("10001".to_string()),
            ),
        ]),
    );

    let data = outer.build();
    let view = JsonbView::new(&data).unwrap();

    let addr = view.get("address").unwrap().unwrap();
    if let JsonbValue::Object(addr_view) = addr {
        let city = addr_view.get("city").unwrap().unwrap();
        assert_eq!(city, JsonbValue::String("NYC"));
    } else {
        panic!("expected object");
    }
}

#[test]
fn jsonb_get_path() {
    let mut outer = JsonbBuilder::new_object();
    outer.set(
        "user",
        JsonbBuilderValue::Object(vec![
            (
                "name".to_string(),
                JsonbBuilderValue::String("Dave".to_string()),
            ),
            (
                "profile".to_string(),
                JsonbBuilderValue::Object(vec![(
                    "email".to_string(),
                    JsonbBuilderValue::String("dave@example.com".to_string()),
                )]),
            ),
        ]),
    );

    let data = outer.build();
    let view = JsonbView::new(&data).unwrap();

    let email = view
        .get_path(&["user", "profile", "email"])
        .unwrap()
        .unwrap();
    assert_eq!(email, JsonbValue::String("dave@example.com"));

    assert!(view.get_path(&["user", "missing"]).unwrap().is_none());
}

#[test]
fn jsonb_null_value() {
    let mut builder = JsonbBuilder::new_object();
    builder.set("value", JsonbBuilderValue::Null);

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    let value = view.get("value").unwrap().unwrap();
    assert_eq!(value, JsonbValue::Null);
}

#[test]
fn jsonb_scalar_types() {
    let null_data = JsonbBuilder::new_null().build();
    let null_view = JsonbView::new(&null_data).unwrap();
    assert_eq!(null_view.as_value().unwrap(), JsonbValue::Null);

    let bool_data = JsonbBuilder::new_bool(true).build();
    let bool_view = JsonbView::new(&bool_data).unwrap();
    assert_eq!(bool_view.as_value().unwrap(), JsonbValue::Bool(true));

    let num_data = JsonbBuilder::new_number(42.5).build();
    let num_view = JsonbView::new(&num_data).unwrap();
    if let JsonbValue::Number(n) = num_view.as_value().unwrap() {
        assert!((n - 42.5).abs() < 0.0001);
    } else {
        panic!("expected number");
    }

    let str_data = JsonbBuilder::new_string("hello").build();
    let str_view = JsonbView::new(&str_data).unwrap();
    assert_eq!(str_view.as_value().unwrap(), JsonbValue::String("hello"));
}

#[test]
fn jsonb_record_roundtrip() {
    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("data", DataType::Jsonb),
    ]);

    let mut jsonb = JsonbBuilder::new_object();
    jsonb.set("key", "value");
    jsonb.set("count", 42);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 1).unwrap();
    builder.set_jsonb(1, &jsonb).unwrap();

    let record_data = builder.build().unwrap();
    let view = RecordView::new(&record_data, &schema).unwrap();

    assert_eq!(view.get_int4(0).unwrap(), 1);

    let jsonb_view = view.get_jsonb(1).unwrap();
    let key_val = jsonb_view.get("key").unwrap().unwrap();
    assert_eq!(key_val, JsonbValue::String("value"));

    let count_val = jsonb_view.get("count").unwrap().unwrap();
    if let JsonbValue::Number(n) = count_val {
        assert!((n - 42.0).abs() < 0.0001);
    } else {
        panic!("expected number");
    }
}

#[test]
fn jsonb_opt_returns_none_for_null() {
    let schema = Schema::new(vec![ColumnDef::new("data", DataType::Jsonb)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_null(0);

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    assert!(view.get_jsonb_opt(0).unwrap().is_none());
}

#[test]
fn jsonb_empty_object() {
    let builder = JsonbBuilder::new_object();
    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(view.object_len().unwrap(), 0);
    assert!(view.get("any").unwrap().is_none());
}

#[test]
fn jsonb_empty_array() {
    let builder = JsonbBuilder::new_array();
    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(view.array_len().unwrap(), 0);
    assert!(view.array_get(0).unwrap().is_none());
}

#[test]
fn jsonb_sorted_keys_binary_search() {
    let mut builder = JsonbBuilder::new_object();
    builder.set("zebra", "last");
    builder.set("apple", "first");
    builder.set("middle", "center");

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(
        view.get("apple").unwrap().unwrap(),
        JsonbValue::String("first")
    );
    assert_eq!(
        view.get("middle").unwrap().unwrap(),
        JsonbValue::String("center")
    );
    assert_eq!(
        view.get("zebra").unwrap().unwrap(),
        JsonbValue::String("last")
    );
}

#[test]
fn jsonb_array_with_mixed_types() {
    let mut builder = JsonbBuilder::new_array();
    builder.push(JsonbBuilderValue::Null);
    builder.push(true);
    builder.push(42);
    builder.push("text");

    let data = builder.build();
    let view = JsonbView::new(&data).unwrap();

    assert_eq!(view.array_len().unwrap(), 4);
    assert_eq!(view.array_get(0).unwrap().unwrap(), JsonbValue::Null);
    assert_eq!(view.array_get(1).unwrap().unwrap(), JsonbValue::Bool(true));
}

#[test]
fn data_type_interval_fixed_size() {
    assert_eq!(DataType::Interval.fixed_size(), Some(16));
}

#[test]
fn interval_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("duration", DataType::Interval)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_interval(0, 3600_000_000, 5, 2).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let (micros, days, months) = view.get_interval(0).unwrap();
    assert_eq!(micros, 3600_000_000);
    assert_eq!(days, 5);
    assert_eq!(months, 2);
}

#[test]
fn data_type_enum_fixed_size() {
    assert_eq!(DataType::Enum.fixed_size(), Some(4));
}

#[test]
fn enum_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("status", DataType::Enum)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_enum(0, 1, 42).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let (type_id, ordinal) = view.get_enum(0).unwrap();
    assert_eq!(type_id, 1);
    assert_eq!(ordinal, 42);
}

#[test]
fn data_type_point_fixed_size() {
    assert_eq!(DataType::Point.fixed_size(), Some(16));
}

#[test]
fn point_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("location", DataType::Point)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_point(0, 1.5, 2.5).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let (x, y) = view.get_point(0).unwrap();
    assert!((x - 1.5).abs() < 0.0001);
    assert!((y - 2.5).abs() < 0.0001);
}

#[test]
fn data_type_box_fixed_size() {
    assert_eq!(DataType::Box.fixed_size(), Some(32));
}

#[test]
fn box_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("bounds", DataType::Box)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_box(0, (0.0, 0.0), (10.0, 20.0)).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let ((lx, ly), (hx, hy)) = view.get_box(0).unwrap();
    assert!((lx - 0.0).abs() < 0.0001);
    assert!((ly - 0.0).abs() < 0.0001);
    assert!((hx - 10.0).abs() < 0.0001);
    assert!((hy - 20.0).abs() < 0.0001);
}

#[test]
fn data_type_circle_fixed_size() {
    assert_eq!(DataType::Circle.fixed_size(), Some(24));
}

#[test]
fn circle_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("area", DataType::Circle)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_circle(0, (5.0, 5.0), 3.0).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let ((cx, cy), radius) = view.get_circle(0).unwrap();
    assert!((cx - 5.0).abs() < 0.0001);
    assert!((cy - 5.0).abs() < 0.0001);
    assert!((radius - 3.0).abs() < 0.0001);
}

#[test]
fn data_type_int4_range_fixed_size() {
    assert_eq!(DataType::Int4Range.fixed_size(), Some(9));
}

#[test]
fn int4_range_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("range", DataType::Int4Range)]);

    let mut builder = RecordBuilder::new(&schema);
    builder
        .set_int4_range(0, Some(10), Some(20), true, false)
        .unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let range = view.get_int4_range(0).unwrap();
    assert_eq!(range.lower, Some(10));
    assert_eq!(range.upper, Some(20));
    assert!(range.lower_inclusive);
    assert!(!range.upper_inclusive);
    assert!(!range.is_empty);
}

#[test]
fn int4_range_empty() {
    let schema = Schema::new(vec![ColumnDef::new("range", DataType::Int4Range)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4_range_empty(0).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let range = view.get_int4_range(0).unwrap();
    assert!(range.is_empty);
}

#[test]
fn data_type_int8_range_fixed_size() {
    assert_eq!(DataType::Int8Range.fixed_size(), Some(17));
}

#[test]
fn int8_range_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("range", DataType::Int8Range)]);

    let mut builder = RecordBuilder::new(&schema);
    builder
        .set_int8_range(0, Some(100), Some(200), true, true)
        .unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let range = view.get_int8_range(0).unwrap();
    assert_eq!(range.lower, Some(100));
    assert_eq!(range.upper, Some(200));
    assert!(range.lower_inclusive);
    assert!(range.upper_inclusive);
}

#[test]
fn data_type_date_range_fixed_size() {
    assert_eq!(DataType::DateRange.fixed_size(), Some(9));
}

#[test]
fn date_range_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("period", DataType::DateRange)]);

    let mut builder = RecordBuilder::new(&schema);
    builder
        .set_date_range(0, Some(19000), Some(19365), true, false)
        .unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let range = view.get_date_range(0).unwrap();
    assert_eq!(range.lower, Some(19000));
    assert_eq!(range.upper, Some(19365));
}

#[test]
fn data_type_timestamp_range_fixed_size() {
    assert_eq!(DataType::TimestampRange.fixed_size(), Some(17));
}

#[test]
fn timestamp_range_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("period", DataType::TimestampRange)]);

    let mut builder = RecordBuilder::new(&schema);
    builder
        .set_timestamp_range(0, Some(1000000), Some(2000000), true, false)
        .unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let range = view.get_timestamp_range(0).unwrap();
    assert_eq!(range.lower, Some(1000000));
    assert_eq!(range.upper, Some(2000000));
}

#[test]
fn data_type_decimal_is_variable() {
    assert!(DataType::Decimal.is_variable());
}

#[test]
fn decimal_roundtrip() {
    let schema = Schema::new(vec![ColumnDef::new("amount", DataType::Decimal)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_decimal(0, 12345, 2, false).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let decimal = view.get_decimal(0).unwrap();
    assert_eq!(decimal.digits(), 12345);
    assert_eq!(decimal.scale(), 2);
    assert!(!decimal.is_negative());
}

#[test]
fn decimal_negative_value() {
    let schema = Schema::new(vec![ColumnDef::new("amount", DataType::Decimal)]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_decimal(0, 99999, 4, true).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    let decimal = view.get_decimal(0).unwrap();
    assert!(decimal.is_negative());
    assert_eq!(decimal.scale(), 4);
}

#[test]
fn array_builder_creates_int4_array() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Int4);
    builder.push_int4(10);
    builder.push_int4(20);
    builder.push_int4(30);

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.len(), 3);
    assert_eq!(view.elem_type(), DataType::Int4);
    assert_eq!(view.get_int4(0).unwrap(), 10);
    assert_eq!(view.get_int4(1).unwrap(), 20);
    assert_eq!(view.get_int4(2).unwrap(), 30);
}

#[test]
fn array_view_returns_none_for_out_of_bounds() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Int4);
    builder.push_int4(42);

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert!(view.get_int4(0).is_ok());
    assert!(view.get_int4(1).is_err());
}

#[test]
fn array_builder_creates_text_array() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Text);
    builder.push_text("hello");
    builder.push_text("world");
    builder.push_text("!");

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.len(), 3);
    assert_eq!(view.elem_type(), DataType::Text);
    assert_eq!(view.get_text(0).unwrap(), "hello");
    assert_eq!(view.get_text(1).unwrap(), "world");
    assert_eq!(view.get_text(2).unwrap(), "!");
}

#[test]
fn array_view_empty_array() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let builder = ArrayBuilder::new(DataType::Int4);
    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.len(), 0);
    assert_eq!(view.elem_type(), DataType::Int4);
}

#[test]
fn array_builder_handles_nulls() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Int4);
    builder.push_int4(10);
    builder.push_null();
    builder.push_int4(30);

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.len(), 3);
    assert!(!view.is_null(0));
    assert!(view.is_null(1));
    assert!(!view.is_null(2));
    assert_eq!(view.get_int4(0).unwrap(), 10);
    assert_eq!(view.get_int4(2).unwrap(), 30);
}

#[test]
fn array_view_zero_copy_text_access() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Text);
    builder.push_text("test_string");

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    let text = view.get_text(0).unwrap();
    let text_ptr = text.as_ptr();
    let data_ptr = data.as_ptr();
    assert!(text_ptr >= data_ptr && text_ptr < unsafe { data_ptr.add(data.len()) });
}

#[test]
fn array_builder_float8_array() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Float8);
    builder.push_float8(1.5);
    builder.push_float8(2.5);

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.len(), 2);
    assert!((view.get_float8(0).unwrap() - 1.5).abs() < 0.0001);
    assert!((view.get_float8(1).unwrap() - 2.5).abs() < 0.0001);
}

#[test]
fn array_view_fixed_type_o1_access() {
    use crate::records::array::{ArrayBuilder, ArrayView};

    let mut builder = ArrayBuilder::new(DataType::Int8);
    for i in 0..100 {
        builder.push_int8(i * 1000);
    }

    let data = builder.build();
    let view = ArrayView::new(&data).unwrap();

    assert_eq!(view.get_int8(50).unwrap(), 50000);
    assert_eq!(view.get_int8(99).unwrap(), 99000);
    assert_eq!(view.get_int8(0).unwrap(), 0);
}

#[test]
fn data_type_composite_is_variable() {
    assert!(DataType::Composite.is_variable());
    assert_eq!(DataType::Composite.fixed_size(), None);
}

#[test]
fn composite_view_basic_field_access() {
    use crate::records::composite::CompositeView;

    let inner_schema = Schema::new(vec![
        ColumnDef::new("street", DataType::Text),
        ColumnDef::new("city", DataType::Text),
        ColumnDef::new("zip", DataType::Int4),
    ]);

    let mut inner_builder = RecordBuilder::new(&inner_schema);
    inner_builder.set_text(0, "123 Main St").unwrap();
    inner_builder.set_text(1, "Springfield").unwrap();
    inner_builder.set_int4(2, 12345).unwrap();
    let inner_data = inner_builder.build().unwrap();

    let view = CompositeView::new(&inner_data, 3).unwrap();

    assert_eq!(view.field_count(), 3);
    assert!(!view.is_null(0));
    assert!(!view.is_null(1));
    assert!(!view.is_null(2));
}

#[test]
fn composite_view_get_field_returns_data_payload() {
    use crate::records::composite::CompositeView;

    let inner_schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("value", DataType::Int8),
    ]);

    let mut inner_builder = RecordBuilder::new(&inner_schema);
    inner_builder.set_int4(0, 42).unwrap();
    inner_builder.set_int8(1, 123456789).unwrap();
    let inner_data = inner_builder.build().unwrap();

    let view = CompositeView::new(&inner_data, 2).unwrap();

    let payload = view.get_field(0).unwrap();
    assert!(payload.len() >= 12);

    let val0 = i32::from_le_bytes(payload[0..4].try_into().unwrap());
    assert_eq!(val0, 42);

    let val1 = i64::from_le_bytes(payload[4..12].try_into().unwrap());
    assert_eq!(val1, 123456789);
}

#[test]
fn composite_view_handles_null_fields() {
    use crate::records::composite::CompositeView;

    let inner_schema = Schema::new(vec![
        ColumnDef::new("a", DataType::Int4),
        ColumnDef::new("b", DataType::Int4),
    ]);

    let mut inner_builder = RecordBuilder::new(&inner_schema);
    inner_builder.set_int4(0, 100).unwrap();
    inner_builder.set_null(1);
    let inner_data = inner_builder.build().unwrap();

    let view = CompositeView::new(&inner_data, 2).unwrap();

    assert!(!view.is_null(0));
    assert!(view.is_null(1));
}

#[test]
fn composite_view_rejects_empty_data() {
    use crate::records::composite::CompositeView;

    let result = CompositeView::new(&[], 2);
    assert!(result.is_err());
}

#[test]
fn record_builder_set_composite_roundtrip() {
    let inner_schema = Schema::new(vec![
        ColumnDef::new("x", DataType::Int4),
        ColumnDef::new("y", DataType::Int4),
    ]);

    let mut inner_builder = RecordBuilder::new(&inner_schema);
    inner_builder.set_int4(0, 10).unwrap();
    inner_builder.set_int4(1, 20).unwrap();
    let inner_data = inner_builder.build().unwrap();

    let outer_schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("point", DataType::Composite),
    ]);

    let mut outer_builder = RecordBuilder::new(&outer_schema);
    outer_builder.set_int4(0, 1).unwrap();
    outer_builder.set_composite(1, &inner_data).unwrap();

    let outer_data = outer_builder.build().unwrap();
    let outer_view = RecordView::new(&outer_data, &outer_schema).unwrap();

    assert_eq!(outer_view.get_int4(0).unwrap(), 1);

    let composite = outer_view.get_composite(1, 2).unwrap();
    assert_eq!(composite.field_count(), 2);
}

#[test]
fn record_view_get_array_returns_array_view() {
    use crate::records::array::ArrayBuilder;

    let mut arr_builder = ArrayBuilder::new(DataType::Int4);
    arr_builder.push_int4(1);
    arr_builder.push_int4(2);
    arr_builder.push_int4(3);
    let arr_data = arr_builder.build();

    let schema = Schema::new(vec![
        ColumnDef::new("id", DataType::Int4),
        ColumnDef::new("numbers", DataType::Array),
    ]);

    let mut builder = RecordBuilder::new(&schema);
    builder.set_int4(0, 42).unwrap();
    builder.set_array(1, &arr_data).unwrap();

    let data = builder.build().unwrap();
    let view = RecordView::new(&data, &schema).unwrap();

    assert_eq!(view.get_int4(0).unwrap(), 42);

    let arr = view.get_array(1).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get_int4(0).unwrap(), 1);
    assert_eq!(arr.get_int4(1).unwrap(), 2);
    assert_eq!(arr.get_int4(2).unwrap(), 3);
}

#[test]
fn nested_composite_within_composite() {
    use crate::records::composite::CompositeView;

    let address_schema = Schema::new(vec![
        ColumnDef::new("city", DataType::Text),
        ColumnDef::new("zip", DataType::Int4),
    ]);

    let mut addr_builder = RecordBuilder::new(&address_schema);
    addr_builder.set_text(0, "Boston").unwrap();
    addr_builder.set_int4(1, 02101).unwrap();
    let addr_data = addr_builder.build().unwrap();

    let person_schema = Schema::new(vec![
        ColumnDef::new("name", DataType::Text),
        ColumnDef::new("address", DataType::Composite),
    ]);

    let mut person_builder = RecordBuilder::new(&person_schema);
    person_builder.set_text(0, "Alice").unwrap();
    person_builder.set_composite(1, &addr_data).unwrap();
    let person_data = person_builder.build().unwrap();

    let person_view = CompositeView::new(&person_data, 2).unwrap();
    assert_eq!(person_view.field_count(), 2);

    let nested = person_view.get_nested_composite(1, 2).unwrap();
    assert_eq!(nested.field_count(), 2);
}

#[test]
fn composite_view_depth_limit_enforced() {
    use crate::records::composite::{CompositeView, MAX_NESTING_DEPTH};

    let schema = Schema::new(vec![ColumnDef::new("data", DataType::Composite)]);

    let mut data = RecordBuilder::new(&schema);
    data.set_blob(0, &[0x03, 0x00, 0x00]).unwrap();
    let bytes = data.build().unwrap();

    let view = CompositeView::new_with_depth(&bytes, 1, MAX_NESTING_DEPTH - 1).unwrap();
    assert_eq!(view.field_count(), 1);

    let result = view.get_nested_composite(0, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("depth"));
}
