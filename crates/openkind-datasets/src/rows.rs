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
