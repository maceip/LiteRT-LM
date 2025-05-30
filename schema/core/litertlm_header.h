#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_

#include <cstdint>
#include <string>

#include "third_party/flatbuffers/include/flatbuffers/buffer.h"
#include "third_party/flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {

namespace litertlm {

namespace schema {

// LiteRT-LM File Format Version uses Semantic Version Rules (SemVer):
// MAJOR version: increments for incompatible API changes.
// MINOR version: increments on added functionality in a backward
//                compatible manner.
// PATCH version: increments on backward compatible bug fixes.
constexpr uint8_t LITERTLM_MAJOR_VERSION = 1;
constexpr uint8_t LITERTLM_MINOR_VERSION = 0;
constexpr uint8_t LITERTLM_PATCH_VERSION = 0;

// Alias for a fully constructed KeyValuePair for LiteRTLM metadata.
// Users of the CreateKeyValuePair function (see below) will get
// back one of these during the creation of their metadata
// data structures.
using KVPair = ::flatbuffers::Offset<KeyValuePair>;

template <typename T>
struct ValueTypeTraits {
  using SchemaType = T;
};

template <>
struct ValueTypeTraits<uint8_t> {
  using SchemaType = UInt8;
};
template <>
struct ValueTypeTraits<int8_t> {
  using SchemaType = Int8;
};
template <>
struct ValueTypeTraits<uint16_t> {
  using SchemaType = UInt16;
};
template <>
struct ValueTypeTraits<int16_t> {
  using SchemaType = Int16;
};
template <>
struct ValueTypeTraits<uint32_t> {
  using SchemaType = UInt32;
};
template <>
struct ValueTypeTraits<int32_t> {
  using SchemaType = Int32;
};
template <>
struct ValueTypeTraits<float> {
  using SchemaType = Float32;
};
template <>
struct ValueTypeTraits<bool> {
  using SchemaType = Bool;
};
template <>
struct ValueTypeTraits<uint64_t> {
  using SchemaType = UInt64;
};
template <>
struct ValueTypeTraits<int64_t> {
  using SchemaType = Int64;
};

template <typename T>
KVPair CreateKeyValuePair(flatbuffers::FlatBufferBuilder& builder,
                          const std::string& key, const T& value) {
  auto key_offset = builder.CreateString(key);
  typename ValueTypeTraits<T>::SchemaType value_obj(value);
  auto value_offset = builder.CreateStruct(value_obj).Union();

  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value_offset);
  kvp_builder.add_value_type(
      VDataTraits<typename ValueTypeTraits<T>::SchemaType>::enum_value);
  return kvp_builder.Finish();
}

template <>
inline KVPair CreateKeyValuePair(flatbuffers::FlatBufferBuilder& builder,
                                 const std::string& key,
                                 const std::string& value) {
  auto key_offset = builder.CreateString(key);
  // NB: The StringValue object *must* be created before the
  // KeyValuePairBuilder.
  auto value_offset = CreateStringValue(builder, builder.CreateString(value));
  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value_offset.Union());
  kvp_builder.add_value_type(VData::VData_StringValue);
  return kvp_builder.Finish();
}

template <>
inline KVPair CreateKeyValuePair(
    flatbuffers::FlatBufferBuilder& builder, const std::string& key,
    const flatbuffers::Offset<StringValue>& value) {
  auto key_offset = builder.CreateString(key);
  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value.Union());
  kvp_builder.add_value_type(VData::VData_StringValue);
  return kvp_builder.Finish();
}

}  // end namespace schema
}  // end namespace litertlm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_
