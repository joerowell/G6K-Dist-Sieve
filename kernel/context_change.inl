#ifndef INCLUDED_CONTEXTCHANGE_HPP
#error Do not include context_change.inl without context_change.hpp
#endif

constexpr bool operator==(const unsigned lhs,
                          const ContextChange rhs) noexcept {
  return lhs == static_cast<unsigned>(rhs);
}

constexpr bool operator==(const ContextChange lhs,
                          const unsigned rhs) noexcept {
  return rhs == lhs;
}

constexpr bool operator!=(const unsigned lhs,
                          const ContextChange rhs) noexcept {
  return !(lhs == rhs);
}

constexpr bool operator!=(const ContextChange lhs,
                          const unsigned rhs) noexcept {
  return !(lhs == rhs);
}
