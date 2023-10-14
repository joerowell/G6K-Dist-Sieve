#include "thread_pool.hpp"

namespace thread_pool {
thread_local size_t id;
thread_local size_t batch;
} // namespace thread_pool
