#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "fit.h"
#include "processing_functions.h"



__declspec(dllexport) void fit(int ** X, int * y, int X_dim, int y_dim, int ** h, int len_h, int * len_h_i, int * L_h,
	int iteration, double delta, double alpha, int mode_h,
	int len_compact_atribut, int * len_compact_atribut_i,
	double * Pc, double * Ph, double **** Pai_c_h, int ** index_of_hs_atr,double *Convergence)
{
	
	int **h_atr = (int**)malloc(y_dim * sizeof(int*));
	int *len_h_atr_i = (int*)malloc(y_dim * sizeof(int));
	find_h_atr(h_atr, len_h_atr_i, y_dim, h, len_h, len_h_i);
	
	int *L_hs = (int*)malloc(y_dim * sizeof(int));
	find_L_hs(L_hs, y_dim, len_h_atr_i, L_h, h_atr);

	find_Pc(Pc, y, X_dim, y_dim, len_compact_atribut_i);
	
	int L_all_h = find_L_all_h(L_h, len_h);

	srand(time(0));
	start_Ph(Ph, L_all_h);

	double **Ph_i = (double **)malloc(len_h * sizeof(double *));
	start_Ph_i(Ph_i, len_h, L_h);

	start_Pai_c_h(Pai_c_h, len_compact_atribut_i, y_dim, L_hs);
	/*
	Ph[0] = log2(0.2);
	Ph[1] = log2(0.3);
	Ph[2] = log2(0.1);
	Ph[3] = log2(0.4);
	
	Pai_c_h[0][0][0][0]= log2(0.3);
	Pai_c_h[0][0][0][1]= log2(0.7);
	Pai_c_h[0][0][1][0]= log2(0.4);
	Pai_c_h[0][0][1][1]= log2(0.6);
	Pai_c_h[0][0][2][0]= log2(0.5);
	Pai_c_h[0][0][2][1]= log2(0.5);
	Pai_c_h[0][0][3][0]= log2(0.8);
	Pai_c_h[0][0][3][1]= log2(0.2);
	
	Pai_c_h[0][1][0][0]= log2(0.4);
	Pai_c_h[0][1][1][0]= log2(0.5);
	Pai_c_h[0][1][0][1]= log2(0.6);
	Pai_c_h[0][1][1][1]= log2(0.5);
	
	Pai_c_h[0][2][0][0]= log2(0.8);
	Pai_c_h[0][2][0][1]= log2(0.2);
	Pai_c_h[0][2][1][0]= log2(0.6);
	Pai_c_h[0][2][1][1]= log2(0.4);
	
	
	
	Pai_c_h[1][0][0][0]= log2(0.4);
	Pai_c_h[1][0][0][1]= log2(0.6);
	Pai_c_h[1][0][1][0]= log2(0.5);
	Pai_c_h[1][0][1][1]= log2(0.5);
	Pai_c_h[1][0][2][0]= log2(0.6);
	Pai_c_h[1][0][2][1]= log2(0.4);
	Pai_c_h[1][0][3][0]= log2(0.7);
	Pai_c_h[1][0][3][1]= log2(0.3);
	
	Pai_c_h[1][1][0][0]= log2(0.5);
	Pai_c_h[1][1][0][1]= log2(0.5);
	Pai_c_h[1][1][1][0]= log2(0.4);
	Pai_c_h[1][1][1][1]= log2(0.6);
	
	Pai_c_h[1][2][0][0]= log2(0.7);
	Pai_c_h[1][2][0][1]= log2(0.3);
	Pai_c_h[1][2][1][0]= log2(0.8);
	Pai_c_h[1][2][1][1]= log2(0.2);
	*/
	
	sotr_X(X, y, X_dim, y_dim);

	int **new_data;
	int *data_repeat;
	int len_new_data;
	find_new_data(&new_data, &data_repeat, &len_new_data, X, y, X_dim, y_dim);
	
	int **interval = (int **)malloc((y_dim + 1) * sizeof(int *));
	int *len_interval = (int *)malloc((y_dim + 1) * sizeof(int));
	find_interval(interval, len_interval, new_data, len_new_data, y_dim);




	int ***factor = (int ***)malloc(y_dim * sizeof(int **));
	int *len_i_factor = (int *)malloc(y_dim * sizeof(int));
	int **len_ij_factor = (int **)malloc(y_dim * sizeof(int *));

	int **class_factor = (int **)malloc(y_dim * sizeof(int *));
	int *len_i_class_factor = (int *)malloc(y_dim * sizeof(int));
	find_factor(factor, len_i_factor, len_ij_factor, class_factor, len_i_class_factor, y_dim,
		new_data, len_new_data, interval, len_interval);

	
	double **temp_factor = build_temp_factor(y_dim, len_interval);
	
	double **Ex = build_Ex(len_new_data, L_all_h);
	
	int **hs_of_index = build_hs_of_index(L_all_h, L_h, len_h);

	build_index_of_hs_atr(index_of_hs_atr, y_dim, h_atr, len_h_atr_i, hs_of_index, L_all_h, L_h);
	
	
	double ****Pai_c_h_shadow;
	double *Ph_shadow;
	build_shadows(&Pai_c_h_shadow,Pai_c_h, len_compact_atribut_i, y_dim, L_hs,&Ph_shadow,L_all_h,Ph);
	
	
	
	int iterat;
	for (iterat = 0; iterat < iteration; iterat++)
	{

		
		iterat_Ex(L_all_h, len_i_factor, class_factor, len_ij_factor,
			temp_factor, Pai_c_h, index_of_hs_atr, factor,
			y_dim, Ph, Ex, data_repeat, len_new_data,alpha);

		
		iterat_Ph(L_all_h, index_of_hs_atr, Ph, Ex,
			len_new_data, len_h, mode_h, Ph_i, L_h, hs_of_index);

		
		iterat_Pai_c_h(len_compact_atribut_i, index_of_hs_atr, Ex,
			len_new_data, y_dim, L_hs, Pai_c_h, new_data,
			L_all_h, delta*X_dim);

		Convergence[iterat] = Record_Convergence(Pai_c_h_shadow,Pai_c_h,
		 len_compact_atribut_i, y_dim, L_hs,Ph_shadow,L_all_h,Ph);
		
	}
	
	
	
	free_all(hs_of_index, Ex, temp_factor,
		len_i_class_factor, class_factor, len_ij_factor, len_i_factor,
		factor, len_interval, interval, len_new_data, data_repeat,
		new_data, L_all_h, L_hs, len_h_atr_i, h_atr, y_dim,Pai_c_h_shadow,
		Ph_shadow,len_compact_atribut_i,len_h,Ph_i);
	
}



__declspec(dllexport) void continue_(int ** X, int * y, int X_dim, int y_dim, int ** h, int len_h, int * len_h_i, int * L_h,
	int iteration, double delta, double alpha, int mode_h,
	int len_compact_atribut, int * len_compact_atribut_i,
	double * Pc, double * Ph, double **** Pai_c_h, int ** index_of_hs_atr,double *Convergence)
{
	
	int **h_atr = (int**)malloc(y_dim * sizeof(int*));
	int *len_h_atr_i = (int*)malloc(y_dim * sizeof(int));
	find_h_atr(h_atr, len_h_atr_i, y_dim, h, len_h, len_h_i);
	
	int *L_hs = (int*)malloc(y_dim * sizeof(int));
	find_L_hs(L_hs, y_dim, len_h_atr_i, L_h, h_atr);

	find_Pc(Pc, y, X_dim, y_dim, len_compact_atribut_i);
	
	int L_all_h = find_L_all_h(L_h, len_h);



	double **Ph_i = (double **)malloc(len_h * sizeof(double *));
	start_Ph_i(Ph_i, len_h, L_h);

	
	
	sotr_X(X, y, X_dim, y_dim);

	int **new_data;
	int *data_repeat;
	int len_new_data;
	find_new_data(&new_data, &data_repeat, &len_new_data, X, y, X_dim, y_dim);
	
	int **interval = (int **)malloc((y_dim + 1) * sizeof(int *));
	int *len_interval = (int *)malloc((y_dim + 1) * sizeof(int));
	find_interval(interval, len_interval, new_data, len_new_data, y_dim);




	int ***factor = (int ***)malloc(y_dim * sizeof(int **));
	int *len_i_factor = (int *)malloc(y_dim * sizeof(int));
	int **len_ij_factor = (int **)malloc(y_dim * sizeof(int *));

	int **class_factor = (int **)malloc(y_dim * sizeof(int *));
	int *len_i_class_factor = (int *)malloc(y_dim * sizeof(int));
	find_factor(factor, len_i_factor, len_ij_factor, class_factor, len_i_class_factor, y_dim,
		new_data, len_new_data, interval, len_interval);

	
	double **temp_factor = build_temp_factor(y_dim, len_interval);
	
	double **Ex = build_Ex(len_new_data, L_all_h);
	
	int **hs_of_index = build_hs_of_index(L_all_h, L_h, len_h);

	build_index_of_hs_atr(index_of_hs_atr, y_dim, h_atr, len_h_atr_i, hs_of_index, L_all_h, L_h);
	
	
	double ****Pai_c_h_shadow;
	double *Ph_shadow;
	build_shadows(&Pai_c_h_shadow,Pai_c_h, len_compact_atribut_i, y_dim, L_hs,&Ph_shadow,L_all_h,Ph);
	
	
	
	int iterat;
	for (iterat = 0; iterat < iteration; iterat++)
	{

		
		iterat_Ex(L_all_h, len_i_factor, class_factor, len_ij_factor,
			temp_factor, Pai_c_h, index_of_hs_atr, factor,
			y_dim, Ph, Ex, data_repeat, len_new_data,alpha);

		
		iterat_Ph(L_all_h, index_of_hs_atr, Ph, Ex,
			len_new_data, len_h, mode_h, Ph_i, L_h, hs_of_index);

		
		iterat_Pai_c_h(len_compact_atribut_i, index_of_hs_atr, Ex,
			len_new_data, y_dim, L_hs, Pai_c_h, new_data,
			L_all_h, delta*X_dim);

		Convergence[iterat] = Record_Convergence(Pai_c_h_shadow,Pai_c_h,
		 len_compact_atribut_i, y_dim, L_hs,Ph_shadow,L_all_h,Ph);
		
	}
	
	
	
	free_all(hs_of_index, Ex, temp_factor,
		len_i_class_factor, class_factor, len_ij_factor, len_i_factor,
		factor, len_interval, interval, len_new_data, data_repeat,
		new_data, L_all_h, L_hs, len_h_atr_i, h_atr, y_dim,Pai_c_h_shadow,
		Ph_shadow,len_compact_atribut_i,len_h,Ph_i);
	
}

