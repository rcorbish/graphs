#pragma once
#include <stdlib.h>


void eigs( 
        double * h_S,   // output eigenvalues
        double * h_V,   // output eigen vectors
        double * h_A,   // input matrix ( symmetric = upper )
        const int64_t m // m x m matrix
    ) ;

void gram( 
        double * h_A,
        double * h_C,
        const int64_t m,
        const int64_t n
    ) ;
int svd( 
        double * h_U,
        double * h_S,
        double * h_V,
        double * h_A,
        const int64_t m,
        const int64_t n,
        const int64_t lda,
        const int64_t rank
    ) ;

