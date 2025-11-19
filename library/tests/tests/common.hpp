#ifndef MPI_ADVANCE_TEST_ALL_MATRIX_HPP
#define MPI_ADVANCE_TEST_ALL_MATRIX_HPP

void test_matrix(const char*);

void test_all_matrices()
{
    test_matrix("../../../test_data/dwt_162.pm");
    test_matrix("../../../test_data/odepa400.pm");
    test_matrix("../../../test_data/ww_36_pmec_36.pm");
    test_matrix("../../../test_data/bcsstk01.pm");
    test_matrix("../../../test_data/west0132.pm");
    test_matrix("../../../test_data/gams10a.pm");
    test_matrix("../../../test_data/gams10am.pm");
    test_matrix("../../../test_data/D_10.pm");
    test_matrix("../../../test_data/oscil_dcop_11.pm");
    test_matrix("../../../test_data/tumorAntiAngiogenesis_4.pm");
    test_matrix("../../../test_data/ch5-5-b1.pm");
    test_matrix("../../../test_data/msc01050.pm");
    test_matrix("../../../test_data/SmaGri.pm");
    test_matrix("../../../test_data/radfr1.pm");
    test_matrix("../../../test_data/bibd_49_3.pm");
    test_matrix("../../../test_data/can_1054.pm");
    test_matrix("../../../test_data/can_1072.pm");
    test_matrix("../../../test_data/lp_sctap2.pm");
    test_matrix("../../../test_data/lp_woodw.pm");
}

#endif