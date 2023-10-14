#ifndef INCLUDED_QENTRY_HPP
#define INCLUDED_QENTRY_HPP
#include <cstddef>
#include <cstdint>

struct QEntry {
  std::size_t i, j;
  float len;
  std::int8_t sign;
};

#endif
