//! Parquet row reading for installed dataset shards.
//!
//! Rows are materialized as plain JSON maps so benchmark materializers stay
//! independent of Arrow types. Only the column types these datasets actually
//! use are supported; anything else fails with the type name instead of a
//! silent coercion.

use std::fs::File;
use std::path::Path;

use arrow_array::Array;
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    ListArray, StringArray, StructArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

use crate::{Error, Result};

/// Read every row of one parquet shard as a JSON object keyed by column name.
///
/// # Errors
/// Returns an error for parquet decode failures and for unsupported column
/// types, naming the offending type.
pub fn read_parquet_rows(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    let file = File::open(path)?;
    let reader: ParquetRecordBatchReader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| Error::Parquet(format!("open {}: {error}", path.display())))?
        .build()
        .map_err(|error| Error::Parquet(format!("read {}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch =
            batch.map_err(|error| Error::Parquet(format!("decode {}: {error}", path.display())))?;
        let columns = batch.columns();
        let schema = batch.schema();
        let fields = schema.fields();
        for row in 0..batch.num_rows() {
            let mut object = serde_json::Map::new();
            for (column, field) in columns.iter().zip(fields.iter()) {
                object.insert(field.name().to_owned(), value_at(column.as_ref(), row)?);
            }
            rows.push(object);
        }
    }
    Ok(rows)
}

fn value_at(array: &dyn Array, row: usize) -> Result<serde_json::Value> {
    if array.is_null(row) {
        return Ok(serde_json::Value::Null);
    }
    let value = match array.data_type() {
        arrow_schema::DataType::Boolean => {
            serde_json::Value::Bool(downcast::<BooleanArray>(array)?.value(row))
        }
        arrow_schema::DataType::Int8 => number(downcast::<Int8Array>(array)?.value(row)),
        arrow_schema::DataType::Int16 => number(downcast::<Int16Array>(array)?.value(row)),
        arrow_schema::DataType::Int32 => number(downcast::<Int32Array>(array)?.value(row)),
        arrow_schema::DataType::Int64 => number(downcast::<Int64Array>(array)?.value(row)),
        arrow_schema::DataType::UInt8 => {
            serde_json::json!(downcast::<UInt8Array>(array)?.value(row))
        }
        arrow_schema::DataType::UInt16 => {
            serde_json::json!(downcast::<UInt16Array>(array)?.value(row))
        }
        arrow_schema::DataType::UInt32 => {
            serde_json::json!(downcast::<UInt32Array>(array)?.value(row))
        }
        arrow_schema::DataType::UInt64 => {
            serde_json::json!(downcast::<UInt64Array>(array)?.value(row))
        }
        arrow_schema::DataType::Float32 => {
            serde_json::json!(downcast::<Float32Array>(array)?.value(row))
        }
        arrow_schema::DataType::Float64 => {
            serde_json::json!(downcast::<Float64Array>(array)?.value(row))
        }
        arrow_schema::DataType::Utf8 => {
            serde_json::Value::String(downcast::<StringArray>(array)?.value(row).to_owned())
        }
        arrow_schema::DataType::List(_) => {
            let list = downcast::<ListArray>(array)?;
            let item = list.value(row);
            let mut values = Vec::with_capacity(item.len());
            for index in 0..item.len() {
                values.push(value_at(item.as_ref(), index)?);
            }
            serde_json::Value::Array(values)
        }
        arrow_schema::DataType::Struct(fields) => {
            let structure = downcast::<StructArray>(array)?;
            let slice = structure.slice(row, 1);
            let mut object = serde_json::Map::new();
            for (column, field) in slice.columns().iter().zip(fields.iter()) {
                object.insert(field.name().to_owned(), value_at(column.as_ref(), 0)?);
            }
            serde_json::Value::Object(object)
        }
        other => return Err(Error::Parquet(format!("unsupported column type {other:?}"))),
    };
    Ok(value)
}

fn downcast<A: Array + 'static>(array: &dyn Array) -> Result<&A> {
    array
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| Error::Parquet("column type changed between rows".into()))
}

fn number(value: impl Into<i64>) -> serde_json::Value {
    serde_json::json!(value.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int8Array, ListArray, RecordBatch, StringArray, StructArray, UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field, Fields};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_batch(path: &Path, batch: &RecordBatch) {
        let file = File::create(path).expect("create shard");
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("writer");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
    }

    #[test]
    fn parquet_rows_materialize_every_supported_column_type() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("shard.parquet");

        let values_field = Field::new("item", DataType::Int64, true);
        let list =
            ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>(vec![Some(
                vec![Some(1_i64), Some(2), Some(3)],
            )]);
        let struct_fields = Fields::from(vec![
            Field::new("inner", DataType::Utf8, false),
            Field::new("flag", DataType::Boolean, false),
        ]);
        let structure = StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["inner-value"])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("i8", DataType::Int8, true),
                Field::new("i16", DataType::Int16, false),
                Field::new("i32", DataType::Int32, false),
                Field::new("i64", DataType::Int64, false),
                Field::new("u8", DataType::UInt8, false),
                Field::new("u64", DataType::UInt64, false),
                Field::new("f32", DataType::Float32, false),
                Field::new("f64", DataType::Float64, false),
                Field::new("flag", DataType::Boolean, false),
                Field::new("tags", DataType::List(Arc::new(values_field)), false),
                Field::new("meta", DataType::Struct(struct_fields), true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["row-a"])),
                Arc::new(Int8Array::from(vec![None])),
                Arc::new(Int16Array::from(vec![-2_i16])),
                Arc::new(Int32Array::from(vec![-3_i32])),
                Arc::new(Int64Array::from(vec![4_i64])),
                Arc::new(UInt8Array::from(vec![5_u8])),
                Arc::new(UInt64Array::from(vec![6_u64])),
                Arc::new(Float32Array::from(vec![1.5_f32])),
                Arc::new(Float64Array::from(vec![2.5_f64])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(list),
                Arc::new(structure),
            ],
        )
        .expect("batch");

        write_batch(&path, &batch);
        let rows = read_parquet_rows(&path).expect("read rows");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["text"], "row-a");
        assert_eq!(row["i8"], serde_json::Value::Null);
        assert_eq!(row["i16"], -2);
        assert_eq!(row["i32"], -3);
        assert_eq!(row["i64"], 4);
        assert_eq!(row["u8"], 5);
        assert_eq!(row["u64"], 6);
        assert_eq!(row["f32"], 1.5);
        assert_eq!(row["f64"], 2.5);
        assert_eq!(row["flag"], false);
        assert_eq!(row["tags"], serde_json::json!([1, 2, 3]));
        assert_eq!(
            row["meta"],
            serde_json::json!({"inner": "inner-value", "flag": true})
        );
    }

    #[test]
    fn unsupported_column_types_name_the_type() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("shard.parquet");
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "when",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            )])),
            vec![Arc::new(arrow_array::TimestampSecondArray::from(vec![
                1_i64,
            ]))],
        )
        .expect("batch");
        write_batch(&path, &batch);

        let error = read_parquet_rows(&path).expect_err("timestamps are unsupported");
        assert!(
            error.to_string().contains("unsupported column type")
                && error.to_string().contains("Timestamp"),
            "unexpected error: {error}"
        );
    }
}
