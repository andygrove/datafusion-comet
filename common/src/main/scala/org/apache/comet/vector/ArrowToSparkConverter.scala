/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.comet.vector

import org.apache.spark.sql.execution.vectorized.WritableColumnVector
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnVector

/**
 * Utility object for converting Arrow columnar vectors to Spark columnar vectors. This enables
 * caching of columnar data without row conversions.
 */
object ArrowToSparkConverter {

  /**
   * Copies data from a Comet (Arrow) vector to a Spark WritableColumnVector.
   *
   * @param source
   *   The source Comet vector
   * @param target
   *   The target Spark WritableColumnVector
   * @param dataType
   *   The data type of the column
   * @param numRows
   *   The number of rows to copy
   */
  def copyArrowToSparkVector(
      source: ColumnVector,
      target: WritableColumnVector,
      dataType: DataType,
      numRows: Int): Unit = {

    dataType match {
      case BooleanType =>
        copyBooleanColumn(source, target, numRows)

      case ByteType =>
        copyByteColumn(source, target, numRows)

      case ShortType =>
        copyShortColumn(source, target, numRows)

      case IntegerType =>
        copyIntColumn(source, target, numRows)

      case LongType =>
        copyLongColumn(source, target, numRows)

      case FloatType =>
        copyFloatColumn(source, target, numRows)

      case DoubleType =>
        copyDoubleColumn(source, target, numRows)

      case StringType =>
        copyStringColumn(source, target, numRows)

      case BinaryType =>
        copyBinaryColumn(source, target, numRows)

      case dt: DecimalType =>
        copyDecimalColumn(source, target, dt, numRows)

      case DateType =>
        copyDateColumn(source, target, numRows)

      case TimestampType =>
        copyTimestampColumn(source, target, numRows)

      case ArrayType(elementType, containsNull) =>
        copyArrayColumn(source, target, elementType, containsNull, numRows)

      case StructType(fields) =>
        copyStructColumn(source, target, fields, numRows)

      case MapType(keyType, valueType, valueContainsNull) =>
        copyMapColumn(source, target, keyType, valueType, valueContainsNull, numRows)

      case _ =>
        throw new UnsupportedOperationException(s"Unsupported data type: $dataType")
    }
  }

  private def copyBooleanColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        target.putBoolean(i, source.getBoolean(i))
      }
    }
  }

  private def copyByteColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        target.putByte(i, source.getByte(i))
      }
    }
  }

  private def copyShortColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        target.putShort(i, source.getShort(i))
      }
    }
  }

  private def copyIntColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    // Try bulk copy for better performance
    if (source.hasNull) {
      for (i <- 0 until numRows) {
        if (source.isNullAt(i)) {
          target.putNull(i)
        } else {
          target.putInt(i, source.getInt(i))
        }
      }
    } else {
      // No nulls, can do bulk copy
      val intArray = new Array[Int](numRows)
      for (i <- 0 until numRows) {
        intArray(i) = source.getInt(i)
      }
      target.putInts(0, numRows, intArray, 0)
    }
  }

  private def copyLongColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    if (source.hasNull) {
      for (i <- 0 until numRows) {
        if (source.isNullAt(i)) {
          target.putNull(i)
        } else {
          target.putLong(i, source.getLong(i))
        }
      }
    } else {
      // No nulls, can do bulk copy
      val longArray = new Array[Long](numRows)
      for (i <- 0 until numRows) {
        longArray(i) = source.getLong(i)
      }
      target.putLongs(0, numRows, longArray, 0)
    }
  }

  private def copyFloatColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    if (source.hasNull) {
      for (i <- 0 until numRows) {
        if (source.isNullAt(i)) {
          target.putNull(i)
        } else {
          target.putFloat(i, source.getFloat(i))
        }
      }
    } else {
      // No nulls, can do bulk copy
      val floatArray = new Array[Float](numRows)
      for (i <- 0 until numRows) {
        floatArray(i) = source.getFloat(i)
      }
      target.putFloats(0, numRows, floatArray, 0)
    }
  }

  private def copyDoubleColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    if (source.hasNull) {
      for (i <- 0 until numRows) {
        if (source.isNullAt(i)) {
          target.putNull(i)
        } else {
          target.putDouble(i, source.getDouble(i))
        }
      }
    } else {
      // No nulls, can do bulk copy
      val doubleArray = new Array[Double](numRows)
      for (i <- 0 until numRows) {
        doubleArray(i) = source.getDouble(i)
      }
      target.putDoubles(0, numRows, doubleArray, 0)
    }
  }

  private def copyStringColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        val utf8 = source.getUTF8String(i)
        target.putByteArray(i, utf8.getBytes)
      }
    }
  }

  private def copyBinaryColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        target.putByteArray(i, source.getBinary(i))
      }
    }
  }

  private def copyDecimalColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      dt: DecimalType,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        val decimal = source.getDecimal(i, dt.precision, dt.scale)
        target.putDecimal(i, decimal, dt.precision)
      }
    }
  }

  private def copyDateColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    // Dates are stored as integers (days since epoch)
    copyIntColumn(source, target, numRows)
  }

  private def copyTimestampColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      numRows: Int): Unit = {
    // Timestamps are stored as longs (microseconds since epoch)
    copyLongColumn(source, target, numRows)
  }

  private def copyArrayColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      elementType: DataType,
      containsNull: Boolean,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        val sourceArray = source.getArray(i)
        val arrayLength = sourceArray.numElements()

        // Reserve space for the array
        target.putArray(i, 0, arrayLength)

        // Get the child column vector for array elements
        val targetArrayData = target.arrayData()
        val offset = target.getArrayOffset(i)

        // Copy array elements - sourceArray is a ColumnarArray which extends ColumnVector
        for (j <- 0 until arrayLength) {
          if (containsNull && sourceArray.isNullAt(j)) {
            targetArrayData.putNull(offset + j)
          } else {
            copyElementFromArray(sourceArray, j, targetArrayData, offset + j, elementType)
          }
        }
      }
    }
  }

  private def copyStructColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      fields: Array[StructField],
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        val sourceStruct = source.getStruct(i)
        // Copy each field of the struct
        for ((field, fieldIdx) <- fields.zipWithIndex) {
          val targetChild = target.getChild(fieldIdx)
          if (sourceStruct.isNullAt(fieldIdx)) {
            targetChild.putNull(i)
          } else {
            copyElementFromRow(sourceStruct, fieldIdx, targetChild, i, field.dataType)
          }
        }
      }
    }
  }

  private def copyMapColumn(
      source: ColumnVector,
      target: WritableColumnVector,
      keyType: DataType,
      valueType: DataType,
      valueContainsNull: Boolean,
      numRows: Int): Unit = {
    for (i <- 0 until numRows) {
      if (source.isNullAt(i)) {
        target.putNull(i)
      } else {
        val sourceMap = source.getMap(i)
        val mapSize = sourceMap.numElements()

        // Reserve space for the map
        target.putArray(i, 0, mapSize)

        // Get key and value column vectors
        val targetKeys = target.getChild(0)
        val targetValues = target.getChild(1)
        val offset = target.getArrayOffset(i)

        // Copy keys and values
        val keyArray = sourceMap.keyArray()
        val valueArray = sourceMap.valueArray()

        for (j <- 0 until mapSize) {
          // Keys cannot be null in Spark SQL
          copyElementFromArray(keyArray, j, targetKeys, offset + j, keyType)

          if (valueContainsNull && valueArray.isNullAt(j)) {
            targetValues.putNull(offset + j)
          } else {
            copyElementFromArray(valueArray, j, targetValues, offset + j, valueType)
          }
        }
      }
    }
  }

  private def copyElement(
      source: ColumnVector,
      sourceIdx: Int,
      target: WritableColumnVector,
      targetIdx: Int,
      dataType: DataType): Unit = {
    dataType match {
      case BooleanType => target.putBoolean(targetIdx, source.getBoolean(sourceIdx))
      case ByteType => target.putByte(targetIdx, source.getByte(sourceIdx))
      case ShortType => target.putShort(targetIdx, source.getShort(sourceIdx))
      case IntegerType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case LongType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case FloatType => target.putFloat(targetIdx, source.getFloat(sourceIdx))
      case DoubleType => target.putDouble(targetIdx, source.getDouble(sourceIdx))
      case StringType =>
        val utf8 = source.getUTF8String(sourceIdx)
        target.putByteArray(targetIdx, utf8.getBytes)
      case BinaryType => target.putByteArray(targetIdx, source.getBinary(sourceIdx))
      case dt: DecimalType =>
        val decimal = source.getDecimal(sourceIdx, dt.precision, dt.scale)
        target.putDecimal(targetIdx, decimal, dt.precision)
      case DateType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case TimestampType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case _ =>
        throw new UnsupportedOperationException(s"Unsupported element type: $dataType")
    }
  }

  private def copyElementFromArray(
      source: org.apache.spark.sql.vectorized.ColumnarArray,
      sourceIdx: Int,
      target: WritableColumnVector,
      targetIdx: Int,
      dataType: DataType): Unit = {
    dataType match {
      case BooleanType => target.putBoolean(targetIdx, source.getBoolean(sourceIdx))
      case ByteType => target.putByte(targetIdx, source.getByte(sourceIdx))
      case ShortType => target.putShort(targetIdx, source.getShort(sourceIdx))
      case IntegerType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case LongType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case FloatType => target.putFloat(targetIdx, source.getFloat(sourceIdx))
      case DoubleType => target.putDouble(targetIdx, source.getDouble(sourceIdx))
      case StringType =>
        val utf8 = source.getUTF8String(sourceIdx)
        target.putByteArray(targetIdx, utf8.getBytes)
      case BinaryType => target.putByteArray(targetIdx, source.getBinary(sourceIdx))
      case dt: DecimalType =>
        val decimal = source.getDecimal(sourceIdx, dt.precision, dt.scale)
        target.putDecimal(targetIdx, decimal, dt.precision)
      case DateType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case TimestampType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case _ =>
        throw new UnsupportedOperationException(s"Unsupported element type: $dataType")
    }
  }

  private def copyElementFromRow(
      source: org.apache.spark.sql.vectorized.ColumnarRow,
      sourceIdx: Int,
      target: WritableColumnVector,
      targetIdx: Int,
      dataType: DataType): Unit = {
    dataType match {
      case BooleanType => target.putBoolean(targetIdx, source.getBoolean(sourceIdx))
      case ByteType => target.putByte(targetIdx, source.getByte(sourceIdx))
      case ShortType => target.putShort(targetIdx, source.getShort(sourceIdx))
      case IntegerType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case LongType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case FloatType => target.putFloat(targetIdx, source.getFloat(sourceIdx))
      case DoubleType => target.putDouble(targetIdx, source.getDouble(sourceIdx))
      case StringType =>
        val utf8 = source.getUTF8String(sourceIdx)
        target.putByteArray(targetIdx, utf8.getBytes)
      case BinaryType => target.putByteArray(targetIdx, source.getBinary(sourceIdx))
      case dt: DecimalType =>
        val decimal = source.getDecimal(sourceIdx, dt.precision, dt.scale)
        target.putDecimal(targetIdx, decimal, dt.precision)
      case DateType => target.putInt(targetIdx, source.getInt(sourceIdx))
      case TimestampType => target.putLong(targetIdx, source.getLong(sourceIdx))
      case _ =>
        throw new UnsupportedOperationException(s"Unsupported element type: $dataType")
    }
  }
}
