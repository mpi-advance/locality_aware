#ifndef MPI_ADVANCE_TEST_ALL_MATRIX_HPP
#define MPI_ADVANCE_TEST_ALL_MATRIX_HPP

#include <string>

void test_matrix(const char*);

void test_all_matrices()
{
    test_matrix((std::string(TEST_DATA_DIR) + "/dwt_162.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/odepa400.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/ww_36_pmec_36.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/bcsstk01.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/west0132.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/gams10a.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/gams10am.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/D_10.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/oscil_dcop_11.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/tumorAntiAngiogenesis_4.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/ch5-5-b1.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/msc01050.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/SmaGri.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/radfr1.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/bibd_49_3.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/can_1054.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/can_1072.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/lp_sctap2.pm").c_str());
    test_matrix((std::string(TEST_DATA_DIR) + "/lp_woodw.pm").c_str());
}

#endif
