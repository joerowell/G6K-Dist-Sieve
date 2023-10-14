#include "layouts.hpp"
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif
#include "fht_lsh.h"
#include "packed_sync_struct.hpp"
#include "siever.h"

#include <array>
#include <cassert>
#include <climits>
#include <mutex>

template <unsigned long items>
static constexpr std::array<int, items> make_single_block_lengths() noexcept {
  std::array<int, items> arr{};
  // std::fill is not constexpr until C++20, and for-each (i.e "for (auto& v :
  // ...)" apparently generates goto, which is not constexpr til C++20).
  for (unsigned i = 0; i < items; i++) {
    arr[i] = 1;
  }

  return arr;
}

static MPI_Datatype build_params_layout() noexcept {
  // Note: it's really very tempting to make this function
  // serialise the params object in an adhoc manner. In practice,
  // this can cause intermittent issues on some systems. We thus
  // serialise the parameters object using a packed layout, since
  // that helps us deal with the extra string length.

  // We serialise everything, except from the threads field and the transaction
  // bulk size.
  constexpr MPI_Aint offsets[]{
      offsetof(SieverParams, reserved_n),
      offsetof(SieverParams, reserved_db_size),
      offsetof(SieverParams, sample_by_sums),
      offsetof(SieverParams, otf_lift),
      offsetof(SieverParams, lift_radius),
      offsetof(SieverParams, lift_unitary_only),
      offsetof(SieverParams, saturation_ratio),
      offsetof(SieverParams, triplesieve_saturation_radius),
      offsetof(SieverParams, bgj1_improvement_db_ratio),
      offsetof(SieverParams, bgj1_resort_ratio),
      offsetof(SieverParams, topology),
      offsetof(SieverParams, dist_threshold),
      offsetof(SieverParams, scale_factor),
      offsetof(SieverParams, bucket_batches),
      offsetof(SieverParams, scratch_buffers)};

  constexpr auto nitems = sizeof(offsets) / sizeof(MPI_Aint);

  // Note; the liberal use of offsetof is to make sure that the changes to the
  // end of SieverParams doesn't screw the serialisation up.
  static_assert(sizeof(SieverParams::dist_threshold) == sizeof(unsigned),
                "Error: type of dist_threshold has changed.");
  static_assert(sizeof(SieverParams::topology) == sizeof(unsigned),
                "Error: type of topology has changed");

  // We pack topology as an unsigned type to make sure that the code compiles
  // nicely.

  // This just describes the types of each element in the pack.
  static std::array<MPI_Datatype, nitems> types{
      Layouts::get_data_type<decltype(SieverParams::reserved_n)>(),
      Layouts::get_data_type<decltype(SieverParams::reserved_db_size)>(),
      Layouts::get_data_type<decltype(SieverParams::sample_by_sums)>(),
      Layouts::get_data_type<decltype(SieverParams::otf_lift)>(),
      Layouts::get_data_type<decltype(SieverParams::lift_radius)>(),
      Layouts::get_data_type<decltype(SieverParams::lift_unitary_only)>(),
      Layouts::get_data_type<decltype(SieverParams::saturation_ratio)>(),
      Layouts::get_data_type<decltype(
          SieverParams::triplesieve_saturation_radius)>(),
      Layouts::get_data_type<decltype(
          SieverParams::bgj1_improvement_db_ratio)>(),
      Layouts::get_data_type<decltype(SieverParams::bgj1_resort_ratio)>(),
      Layouts::get_data_type<unsigned>(),
      Layouts::get_data_type<decltype(SieverParams::dist_threshold)>(),
      Layouts::get_data_type<decltype(SieverParams::scale_factor)>(),
      Layouts::get_data_type<decltype(SieverParams::bucket_batches)>(),
      Layouts::get_data_type<decltype(SieverParams::scratch_buffers)>()};

  // This describes how large each of the offsets space is, in terms of the
  // specified types.
  static_assert(sizeof(SieverParams::dist_threshold) == sizeof(unsigned),
                "Error: type of dist_threshold has changed.");
  static_assert(sizeof(SieverParams::topology) == sizeof(unsigned),
                "Error: type of topology has changed");

  constexpr auto block_lengths = make_single_block_lengths<nitems>();

  MPI_Datatype mpi_params_type{};
  MPI_Type_create_struct(nitems, block_lengths.data(), offsets, types.data(),
                         &mpi_params_type);
  MPI_Type_commit(&mpi_params_type);
  return mpi_params_type;
}

MPI_Datatype Layouts::get_param_layout() noexcept {
  return build_params_layout();
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can send using param_layout", 2) {
  SieverParams start{};
  // G6K never sets this by default.
  start.reserved_n = 100;
  // randomness for the rest is fine.
  start.bdgl_improvement_db_ratio = 1.9;
  start.bgj1_improvement_db_ratio = 1.9;
  start.otf_lift = false;
  start.saturation_ratio = 9;
  // threads is never changed.
  start.threads = 45;
  // simhash_codes_basedir is never changed
  start.simhash_codes_basedir = "test";

  // The type does change.
  start.topology = DistSieverType::ShuffleSiever;
  // As well as the threshold.
  start.dist_threshold = 100;

  // The memory shouldn't change.
  start.memory = 1000;

  // We'll need to free this later.
  auto type = Layouts::get_param_layout();

  if (test_rank == 0) {
    MPI_Send(&start, 1, type, 1, 0, test_comm);
  } else {
    SieverParams out;
    MPI_Recv(&out, 1, type, 0, 0, test_comm, MPI_STATUS_IGNORE);
    CHECK(out.reserved_n == start.reserved_n);
    CHECK(out.bdgl_improvement_db_ratio == start.bdgl_improvement_db_ratio);
    CHECK(out.bgj1_improvement_db_ratio == start.bgj1_improvement_db_ratio);
    CHECK(out.otf_lift == start.otf_lift);
    CHECK(out.saturation_ratio == start.saturation_ratio);
    CHECK(out.threads != start.threads);
    CHECK(out.simhash_codes_basedir != start.simhash_codes_basedir);
    CHECK(out.topology == start.topology);
    CHECK(out.dist_threshold == start.dist_threshold);
    CHECK(out.memory != start.memory);
  }

  // Free the allocated type.
  MPI_Type_free(&type);
}
#endif

static MPI_Datatype build_cc_layout() noexcept {
  // We only ever send two entries.
  MPI_Datatype type{};
  MPI_Type_contiguous(2, MPI_UNSIGNED, &type);
  MPI_Type_commit(&type);
  return type;
}

MPI_Datatype Layouts::get_cc_layout() noexcept { return build_cc_layout(); }

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_cc_layout", 2) {
  // This just checks that we can actually send two unsigned ints
  // using the type.
  std::array<unsigned, 2> arr{5, 3};
  auto type = Layouts::get_cc_layout();

  if (test_rank == 0) {
    MPI_Bcast(arr.data(), 1, type, 0, test_comm);
  } else {
    std::array<unsigned, 2> out;
    MPI_Bcast(out.data(), 1, type, 0, test_comm);
    CHECK(out[0] == arr[0]);
    CHECK(out[1] == arr[1]);
  }
  MPI_Type_free(&type);
}
#endif

static MPI_Datatype get_entry_layout_x_only(const unsigned n) noexcept {
  static_assert(std::is_same_v<ZT, int16_t>, "Error: ZT is no longer int16_t");
  constexpr auto nitems = 1;
  MPI_Datatype entry_layout;
  constexpr int offset = offsetof(Entry, x);
  MPI_Type_create_indexed_block(nitems, static_cast<int>(n), &offset,
                                MPI_INT16_T, &entry_layout);
  MPI_Type_commit(&entry_layout);
  return entry_layout;
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can send entries", 2) {
  constexpr unsigned n = 70;

  Entry e{};
  std::iota(std::begin(e.x), std::begin(e.x) + n, 0);
  auto type = get_entry_layout_x_only(n);

  if (test_rank == 0) {
    MPI_Send(&e, 1, type, 1, 0, test_comm);
  } else {
    Entry e2{};
    MPI_Recv(&e2, 1, type, 0, 0, test_comm, MPI_STATUS_IGNORE);
    CHECK(e2.x == e.x);
  }

  MPI_Type_free(&type);
}
#endif

static MPI_Datatype get_entry_layout_x_and_c(const unsigned n) noexcept {
  static_assert(std::is_same_v<ZT, int16_t>, "Error: ZT is no longer int16_t");
  constexpr auto nitems = 2;

  static constexpr MPI_Aint offsets[2]{
      static_cast<MPI_Aint>(offsetof(Entry, x)),
      static_cast<MPI_Aint>(offsetof(Entry, c))};

  // This can't be constexpr because MPI types may be a void*, which
  // apparently cannot be cast in a constant expression.
  static MPI_Datatype types[]{MPI_INT16_T, MPI_UINT64_T};

  // We never have a value here that is larger than the maximum value we can put
  // into an int, and so this is safe (this would require us to be sieving in
  // above cryptographically relevant dimensions).
  if (n >= std::numeric_limits<int>::max()) {
    __builtin_unreachable();
  }

  const int block_lengths[2]{static_cast<int>(n), XPC_WORD_LEN};

  MPI_Datatype entry_layout{};
  MPI_Type_create_struct(nitems, block_lengths, offsets, types, &entry_layout);
  MPI_Type_commit(&entry_layout);
  return entry_layout;
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can send x and c", 2) {
  constexpr unsigned n = 70;

  Entry e{};
  std::iota(std::begin(e.x), std::begin(e.x) + n, 0);
  std::iota(std::begin(e.c), std::end(e.c), 1);
  auto type = get_entry_layout_x_and_c(n);

  if (test_rank == 0) {
    MPI_Send(&e, 1, type, 1, 0, test_comm);
  } else {
    Entry e2{};
    MPI_Recv(&e2, 1, type, 0, 0, test_comm, MPI_STATUS_IGNORE);
    CHECK(e2.x == e.x);
    CHECK(e2.c == e.c);
  }

  MPI_Type_free(&type);
}
#endif

static MPI_Datatype get_entry_layout_xc_and_yr(const unsigned n) noexcept {
  static_assert(std::is_same_v<ZT, int16_t>, "Error: ZT is no longer int16_t");
  static_assert(std::is_same_v<LFT, float>, "Error: LFT is no longer float");
  static_assert(std::is_same_v<FT, double>, "Error: LFT is no longer double");

  constexpr auto nitems = 4;

  // N.B This sends the length too because recompute_data_for_entry recomputes
  // both yr and the length if either is missing.
  static constexpr MPI_Aint offsets[4]{
      static_cast<MPI_Aint>(offsetof(Entry, yr)),
      static_cast<MPI_Aint>(offsetof(Entry, x)),
      static_cast<MPI_Aint>(offsetof(Entry, c)),
      static_cast<MPI_Aint>(offsetof(Entry, len)),
  };

  // This can't be constexpr because MPI types may be a void*, which
  // apparently cannot be cast in a constant expression.
  static MPI_Datatype types[]{MPI_FLOAT, MPI_INT16_T, MPI_UINT64_T, MPI_DOUBLE};

  // We never have a value here that is larger than the maximum value we can put
  // into an int, and so this is safe (this would require us to be sieving in
  // above cryptographically relevant dimensions).
  if (n >= std::numeric_limits<int>::max()) {
    __builtin_unreachable();
  }

  const int block_lengths[4]{static_cast<int>(n), static_cast<int>(n),
                             XPC_WORD_LEN, 1};

  MPI_Datatype entry_layout{};
  MPI_Type_create_struct(nitems, block_lengths, offsets, types, &entry_layout);
  MPI_Type_commit(&entry_layout);
  return entry_layout;
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can send x, c, yr, len", 2) {
  constexpr unsigned n = 70;

  Entry e{};
  std::iota(std::begin(e.x), std::begin(e.x) + n, 0);
  std::iota(std::begin(e.c), std::end(e.c), 1);
  std::iota(std::begin(e.yr), std::begin(e.yr) + n, 2);
  e.len = 3;

  auto type = get_entry_layout_xc_and_yr(n);

  if (test_rank == 0) {
    MPI_Send(&e, 1, type, 1, 0, test_comm);
  } else {
    Entry e2{};
    MPI_Recv(&e2, 1, type, 0, 0, test_comm, MPI_STATUS_IGNORE);
    CHECK(e2.x == e.x);
    CHECK(e2.c == e.c);
    CHECK(e2.yr == e.yr);
    CHECK(e2.len == e.len);
  }

  MPI_Type_free(&type);
}
#endif

MPI_Datatype Layouts::get_entry_layout(const unsigned n) noexcept {
  constexpr auto layout_spec = Siever::recompute_recv();
  constexpr auto is_x_only =
      (layout_spec == Siever::recompute_all_no_otf_lift());
  constexpr auto is_x_and_c =
      (layout_spec == Siever::recompute_all_but_c_no_otf_lift());
  constexpr auto is_uid_only =
      (layout_spec == Siever::recompute_only_uid_no_otf_lift());

  static_assert(
      is_x_only || is_x_and_c || is_uid_only,
      "Error: requested serialisation for Entry that is not supported.");

  if (is_x_only) {
    return get_entry_layout_x_only(n);
  }

  if (is_x_and_c) {
    return get_entry_layout_x_and_c(n);
  }

  if (is_uid_only) {
    return get_entry_layout_xc_and_yr(n);
  }
}

uint64_t Layouts::get_entry_size(const unsigned n) noexcept {
  constexpr auto layout_spec = Siever::recompute_recv();
  constexpr auto is_x_only =
      (layout_spec == Siever::recompute_all_no_otf_lift());
  constexpr auto is_x_and_c =
      (layout_spec == Siever::recompute_all_but_c_no_otf_lift());
  constexpr auto is_uid_only =
      (layout_spec == Siever::recompute_only_uid_no_otf_lift());

  static_assert(
      is_x_only || is_x_and_c || is_uid_only,
      "Error: requested serialisation for Entry that is not supported.");

  using X_TYPE = decltype(Entry::x)::value_type;
  using Y_TYPE = decltype(Entry::yr)::value_type;
  using CV_TYPE = decltype(Entry::c);
  using FT_TYPE = decltype(Entry::len);

  if (is_x_only) {
    return sizeof(X_TYPE) * n;
  }

  if (is_x_and_c) {
    return sizeof(CV_TYPE) * n;
  }

  if (is_uid_only) {
    return sizeof(X_TYPE) * n + sizeof(Y_TYPE) * n + sizeof(FT_TYPE) +
           sizeof(CV_TYPE);
  }
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can send entry vectors", 2) {
  constexpr auto size = MAX_SIEVING_DIM;
  constexpr unsigned n = 70;

  std::vector<Entry> db(size);
  unsigned cur = 0;
  for (auto &entry : db) {
    std::iota(std::begin(entry.x), std::begin(entry.x) + n, cur);
    ++cur;
  }

  auto type = Layouts::get_entry_vector_layout(n);

  if (test_rank == 0) {
    MPI_Send(db.data(), size, type, 1, 0, test_comm);
  } else {
    std::vector<Entry> db2(size);

    MPI_Recv(db2.data(), size, type, 0, 0, test_comm, MPI_STATUS_IGNORE);
    for (unsigned i = 0; i < size; i++) {
      CHECK(db2[i].x == db[i].x);
    }
  }

  MPI_Type_free(&type);
}
#endif

static MPI_Datatype resize_entry_layout(MPI_Datatype entry_type) noexcept {
  // This creates a type that allows for skips of sizeof(Entry).
  // This is primarily for allowing us to serialise Entries in
  // (say) a std::vector.
  MPI_Datatype vec_type;
  MPI_Aint lb, extent;
  MPI_Type_get_extent(entry_type, &lb, &extent);
  MPI_Type_create_resized(entry_type, lb, sizeof(Entry), &vec_type);
  MPI_Type_commit(&vec_type);
  MPI_Type_free(&entry_type);
  return vec_type;
}

MPI_Datatype
Layouts::get_entry_vector_layout_x_only(const unsigned n) noexcept {
  // N.B this is freed in resize_entry_layout.
  auto entry_type = get_entry_layout_x_only(n);
  return resize_entry_layout(entry_type);
}

MPI_Datatype Layouts::get_entry_vector_layout(const unsigned n) noexcept {
  // N.B this is freed in resize_entry_layout.
  auto entry_type = Layouts::get_entry_layout(n);
  return resize_entry_layout(entry_type);
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_entry_layout_non_contiguous", 3) {
  // This test is to make sure that non-contiguous sends
  // work across 3 ranks.
  constexpr auto size = 128;
  constexpr auto n = 70;

  std::vector<CompressedEntry> cdb(size);
  std::vector<Entry> db(size);
  for (unsigned i = 0; i < size; i++) {
    std::iota(db[i].x.begin(), db[i].x.begin() + n, i);
  }

  // We make it so the cdb has the odd entries in the early
  // entries and the even entries in the latter portion.
  auto even_iter = cdb.begin() + size / 2;
  auto odd_iter = cdb.begin();

  for (unsigned i = 0; i < size / 2; i++) {
    even_iter->i = 2 * i;
    odd_iter->i = 2 * i + 1;
    even_iter++;
    odd_iter++;
  }

  std::vector<Entry> out(size / 2);
  if (test_rank == 0) {
    // We create a new type for each scattered send.
    auto type1 = Layouts::get_entry_layout_non_contiguous(
        cdb.cbegin(), cdb.cbegin() + size / 2, n);
    auto type2 = Layouts::get_entry_layout_non_contiguous(
        cdb.cbegin() + size / 2, cdb.cend(), n);

    // N.B these sends are blocking, but this should be ok
    // as the arrays are quite small.
    MPI_Send(db.data(), 1, type1, 1, 0, test_comm);
    MPI_Send(db.data(), 1, type2, 2, 0, test_comm);
    MPI_Type_free(&type1);
    MPI_Type_free(&type2);
  } else {
    // By contrast, the receiver's type doesn't have any specialty.
    // This is because the type maps are the same, even if the offsets aren't.
    auto recv_type = Layouts::get_entry_vector_layout(n);
    MPI_Recv(out.data(), size / 2, recv_type, 0, 0, test_comm,
             MPI_STATUS_IGNORE);

    // This is just so we start at the right place for each rank in the list.
    const unsigned pos = (test_rank == 1) ? 0 : size / 2;

    // And now we'll check that we got what we expected.
    for (unsigned i = 0; i < out.size(); i++) {
      CHECK(db[cdb[pos + i].i].x == out[i].x);
    }

    MPI_Type_free(&recv_type);
  }
}
#endif

MPI_Datatype
Layouts::get_entry_layout_non_contiguous(const CEIter begin, const CEIter end,
                                         const unsigned n) noexcept {
  auto entry_type = get_entry_vector_layout(n);
  auto ret_type = get_entry_layout_non_contiguous(begin, end, entry_type);
  MPI_Type_free(&entry_type);
  return ret_type;
}

MPI_Datatype
Layouts::get_entry_layout_non_contiguous(const CEIter begin, const CEIter end,
                                         const MPI_Datatype type) noexcept {

  const auto nitems = std::distance(begin, end);
  std::vector<int> offsets(nitems);

  for (unsigned i = 0; i < nitems; i++) {
    offsets[i] = (begin + i)->i;
  }

  MPI_Datatype database_segment{};
  MPI_Type_create_indexed_block(static_cast<int>(nitems), 1, offsets.data(),
                                type, &database_segment);
  MPI_Type_commit(&database_segment);
  return database_segment;
}

static MPI_Datatype
build_layout_from_cdb(const std::vector<CompressedEntry> &cdb,
                      const IntIter begin, const IntIter end, const unsigned n,
                      std::vector<int> &offsets,
                      MPI_Datatype entry_type) noexcept {

  const auto nitems = std::distance(begin, end);
  // N.B Having offsets as an int vector is a pain point: this limits how much
  // data we can send at once. It probably isn't the end of the world all things
  // considered, but it's worth keeping in mind that it exists.
  offsets.resize(nitems);

  for (unsigned i = 0; i < nitems; i++) {
    offsets[i] = cdb[*(begin + i)].i;
  }

  MPI_Datatype database_segment{};
  MPI_Type_create_indexed_block(static_cast<int>(nitems), 1, offsets.data(),
                                entry_type, &database_segment);
  MPI_Type_commit(&database_segment);
  MPI_Type_free(&entry_type);
  return database_segment;
}

MPI_Datatype Layouts::get_entry_layout_from_cdb_x_only(
    const std::vector<CompressedEntry> &cdb, const IntIter begin,
    const IntIter end, const unsigned n, std::vector<int> &offsets) noexcept {

  // N.B this is freed in the next function.
  auto entry_type = get_entry_vector_layout_x_only(n);
  return build_layout_from_cdb(cdb, begin, end, n, offsets, entry_type);
}

MPI_Datatype Layouts::get_entry_layout_from_cdb(
    const std::vector<CompressedEntry> &cdb, const IntIter begin,
    const IntIter end, const unsigned n, std::vector<int> &offsets) noexcept {

  // N.B This is freed in the next function.
  auto entry_type = get_entry_vector_layout(n);
  return build_layout_from_cdb(cdb, begin, end, n, offsets, entry_type);
}

MPI_Datatype
Layouts::get_entry_layout_non_contiguous(const IntIter begin, const IntIter end,
                                         const unsigned n) noexcept {

  const auto nitems = std::distance(begin, end);
  auto entry_type = get_entry_vector_layout(n);

  std::vector<MPI_Aint> offsets(nitems);
  for (unsigned i = 0; i < nitems; i++) {
    // N.B the scaling is because MPI_Aint expects the offsets in
    // bytes.
    offsets[i] = *(begin + i) * sizeof(Entry);
  }

  // We are always sending an entry.
  const std::vector<MPI_Datatype> types(nitems, entry_type);

  // Each of Entry contains exactly 1 vector we want to send.
  const std::vector<int> block_lengths(nitems, 1);

  MPI_Datatype database_segment{};
  MPI_Type_create_struct(static_cast<int>(nitems), block_lengths.data(),
                         offsets.data(), types.data(), &database_segment);
  MPI_Type_commit(&database_segment);
  MPI_Type_free(&entry_type);
  return database_segment;
}

MPI_Datatype Layouts::get_sync_header_type() noexcept {
  MPI_Datatype type;
  static_assert(std::is_same_v<decltype(SyncHeader::trial_count),
                               decltype(SyncHeader::sat_drop)>,
                "Error: type of SyncHeader changed");

  static_assert(sizeof(SyncHeader) == 2 * sizeof(SyncHeader::trial_count),
                "Error: size of SyncHeader has changed");

  // We just serialise the two variables directly as if they were an array.
  constexpr auto nitems = 2;
  MPI_Type_contiguous(
      nitems, Layouts::get_data_type<decltype(SyncHeader::trial_count)>(),
      &type);
  MPI_Type_commit(&type);
  return type;
}

MPI_Datatype Layouts::get_entry_type(const unsigned n) noexcept {
  MPI_Datatype type;
  MPI_Type_contiguous(n, Layouts::get_data_type<ZT>(), &type);
  MPI_Type_commit(&type);
  return type;
}

MPI_Datatype Layouts::get_product_lsh_data_type_aes() noexcept {
  constexpr auto nitems = 5;
  static constexpr MPI_Aint offsets[nitems]{
      static_cast<MPI_Aint>(offsetof(ProductLSHLayout, n)),
      static_cast<MPI_Aint>(offsetof(ProductLSHLayout, blocks)),
      static_cast<MPI_Aint>(offsetof(ProductLSHLayout, code_size)),
      static_cast<MPI_Aint>(offsetof(ProductLSHLayout, multi_hash)),
      static_cast<MPI_Aint>(offsetof(ProductLSHLayout, seed))};

  static std::array<MPI_Datatype, nitems> types{
      Layouts::get_data_type<decltype(ProductLSHLayout::n)>(),
      Layouts::get_data_type<decltype(ProductLSHLayout::blocks)>(),
      Layouts::get_data_type<decltype(ProductLSHLayout::code_size)>(),
      Layouts::get_data_type<decltype(ProductLSHLayout::multi_hash)>(),
      Layouts::get_data_type<decltype(ProductLSHLayout::seed)>()};

  static constexpr auto block_lengths = make_single_block_lengths<nitems>();

  MPI_Datatype type;
  MPI_Type_create_struct(nitems, block_lengths.data(), offsets, types.data(),
                         &type);
  MPI_Type_commit(&type);
  return type;
}
