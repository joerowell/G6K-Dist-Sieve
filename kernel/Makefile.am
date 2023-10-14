lib_LTLIBRARIES=libg6k.la

libg6k_la_LDFLAGS=-no-undefined -version-info @G6K_LT_CURRENT@:@G6K_LT_REVISION@:@G6K_LT_AGE@ $(PTHREAD_CFLAGS)
libg6k_la_LIBADD=$(LIBQD_LIBS) $(MPILIBS)

#AM_CPPFLAGS=-I../doctest

# We need this all to be on one line, setup.py reads this. Yep, it's a HACK
libg6k_la_SOURCES=sieving.cpp control.cpp bgj1_sieve.cpp bdgl_sieve.cpp hk3_sieve.cpp fht_lsh.cpp params.cpp cpuperf.cpp context_change.cpp layouts.cpp mpi_cast.cpp shuffle_siever.cpp siever_types.cpp topology.cpp mpi_obj.cpp mpi_wrapper.cpp dist_siever.cpp communicator.cpp mpi_request.cpp bgj1_bucketing_interface.cpp thread_pool.cpp
include_HEADERS=bucket_pair.hpp compat.hpp cpuperformance.hpp db.inl fht_lsh.h hash_table.inl parallel_algorithms.hpp random.hpp siever.h siever.inl simhash.inl statistics.hpp thread_pool.hpp untemplate.mac context_change.hpp G6K_MPI.hpp layouts.hpp mpi_cast.hpp mpi_obj.hpp mpi_wrapper.hpp shuffle_siever.hpp siever_types.hpp test_utils.hpp topology.hpp dist_siever.hpp dist_siever.hpp communicator.hpp comm_state.hpp mpi_request.hpp bgj1_bucketing_interface.hpp mpi_bandwidth_tracker.hpp mpi_timer.hpp

bin_PROGRAMS= dist_siever
dist_siever_DEPENDENCIES=libg6k.la
dist_siever_SOURCES=dist_sieve_runner.cpp
dist_siever_LDFLAGS= -no-undefined $(PTHREAD_CFLAGS)
dist_siever_LDADD= libg6k.la
