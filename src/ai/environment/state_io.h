#pragma once
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

// Minimal length-prefixed binary I/O for serializing environment state. Used by
// every VirtualEnvironment's serialize/deserialize and by the Rollout's
// save_state/load_state, so the two stay byte-symmetric.
namespace ai::environment::state_io {

template <typename T>
void write_pod(std::ostream &os, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  os.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
T read_pod(std::istream &is) {
  T value;
  is.read(reinterpret_cast<char *>(&value), sizeof(T));
  return value;
}

// Length-prefixed raw bytes (may contain nulls, e.g. an ALEState blob).
inline void write_bytes(std::ostream &os, const std::string &bytes) {
  write_pod<uint64_t>(os, bytes.size());
  os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline std::string read_bytes(std::istream &is) {
  const auto size = read_pod<uint64_t>(is);
  std::string bytes(size, '\0');
  is.read(bytes.data(), static_cast<std::streamsize>(size));
  return bytes;
}

// Length-prefixed vector of a trivially-copyable element type.
template <typename T>
void write_vector(std::ostream &os, const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  write_pod<uint64_t>(os, values.size());
  os.write(reinterpret_cast<const char *>(values.data()),
           static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <typename T>
std::vector<T> read_vector(std::istream &is) {
  const auto size = read_pod<uint64_t>(is);
  std::vector<T> values(size);
  is.read(reinterpret_cast<char *>(values.data()),
          static_cast<std::streamsize>(size * sizeof(T)));
  return values;
}

}  // namespace ai::environment::state_io
