#pragma once



__declspec(dllexport) void fit(int ** X, int * y, int X_dim, int y_dim, int ** h, int len_h, int * len_h_i, int * L_h,
	int iteration, double delta, double alpha, int mode_h,
	int len_compact_atribut, int * len_compact_atribut_i,
	double * Pc, double * Ph, double **** Pai_c_h, int ** index_of_hs_atr,double *Convergence);

__declspec(dllexport) void continue_(int ** X, int * y, int X_dim, int y_dim, int ** h, int len_h, int * len_h_i, int * L_h,
	int iteration, double delta, double alpha, int mode_h,
	int len_compact_atribut, int * len_compact_atribut_i,
	double * Pc, double * Ph, double **** Pai_c_h, int ** index_of_hs_atr,double *Convergence);

