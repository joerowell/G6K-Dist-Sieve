#ifndef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/extensions/doctest_mpi.h>
#endif

#include "dist_siever.hpp"
#include "mpi_cast.hpp"
#include "mpi_wrapper.hpp"
#include <filesystem>
#include <getopt.h>

int main(int argc, char **argv) {
#ifdef MPI_DIST_TEST
  doctest::mpi_init_thread(argc, argv, MPI_THREAD_MULTIPLE);

  doctest::Context ctx;
  ctx.setOption("reporters", "MpiConsoleReporter");
  ctx.setOption("reporters", "MpiFileReporter");
  ctx.setOption("force-colors", true);
  ctx.applyCommandLine(argc, argv);

  int test_result = ctx.run();
  doctest::mpi_finalize();
  return test_result;
#else
  // These are set just for now.
  SieverParams p{};
  // This is never true.
  p.is_cython = false;

  // By default we use all available threads.
  p.threads = 0;
  int thread_support;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
  if (thread_support < MPI_THREAD_FUNNELED) {
    std::cerr
        << "Error: The MPI implementation does not support MPI_THREAD_FUNNELED"
        << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  static option long_options[] = {{"threads", required_argument, NULL, 't'}};

  int c;
  while ((c = getopt_long(argc, argv, "t:", long_options, NULL)) != -1) {
    switch (c) {
    case 't':
      p.threads = std::atoi(optarg);
      break;
    }
  }

  if (p.threads == 0) {
    p.threads = std::thread::hardware_concurrency();
  }

  p.comm = MPI_Cast::mpi_comm_to_uint64(MPI_COMM_WORLD);

  DistSiever dp{p};
  dp.run();
#ifdef MPI_DIST_TEST
  doctest::mpi_finalize();
#endif
#endif
}
