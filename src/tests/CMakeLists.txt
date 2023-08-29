add_executable(test_matching test_matching.cpp)
target_link_libraries(test_matching mpi_advance gtest pthread )
add_test(MatchingTest mpirun -n 16 ./test_matching)

