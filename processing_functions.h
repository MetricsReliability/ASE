#pragma once


void find_h_atr(int **h_atr, int *len_h_atr_i, int y_dim, int ** h, int len_h, int* len_h_i);
void find_L_hs(int *L_hs, int y_dim, int *len_h_atr_i, int * L_h, int **h_atr);
void find_Pc(double * Pc, int * y, int X_dim, int y_dim, int * len_compact_atribut_i);
int find_L_all_h(int *L_h, int len_h);
void start_Ph(double * Ph, int L_all_h);
void start_Ph_i(double **Ph_i, int len_h, int * L_h);
void start_Pai_c_h(double **** Pai_c_h, int * len_compact_atribut_i, int y_dim, int *L_hs);

int determine_smaller(int * X1, int  y1, int * X2, int  y2, int y_dim);
void sotr_X(int ** X, int * y, int X_dim, int y_dim);


void find_new_data(int ***new_data, int **data_repeat, int *len_new_data,
	int ** X, int * y, int X_dim, int y_dim);


int interval_len(int **new_data, int len_new_data, int featur, int start, int end);
void verify_interval(int **new_data, int len_new_data, int featur, int start, int end, int *interval);
void find_interval(int **interval, int *len_interval, int **new_data, int len_new_data, int y_dim);

void find_factor(int ***factor, int *len_i_factor, int **len_ij_factor, int **class_factor,
	int *len_i_class_factor, int y_dim, int **new_data, int len_new_data,
	int **interval, int *len_interval);

double **build_temp_factor(int y_dim, int *len_interval);

double **build_Ex(int len_new_data, int L_all_h);

int *find_h_from_index(int index, int *L_h, int len_h);
int **build_hs_of_index(int L_all_h, int *L_h, int len_h);

int find_index_h(int *h, int *L_h, int len_h);
void build_index_of_hs_atr(int **index_of_hs_atr, int y_dim, int **h_atr, int *len_h_atr_i
	, int **hs_of_index, int L_all_h, int *L_h);

double find_max(double *array, int length);


void build_shadows(double *****Pai_c_h_shadow,double ****Pai_c_h,int *len_compact_atribut_i,int y_dim,
				int *L_hs,double **Ph_shadow,int L_all_h,double *Ph);





void iterat_Ex(int L_all_h, int *len_i_factor, int **class_factor, int **len_ij_factor,
	double **temp_factor, double ****Pai_c_h, int **index_of_hs_atr, int ***factor,
	int y_dim, double *Ph, double **Ex, int *data_repeat, int len_new_data,double alpha);


void iterat_Ph(int L_all_h, int **index_of_hs_atr, double *Ph, double **Ex,
	int len_new_data, int len_h, int mode_h, double **Ph_i, int *L_h, int **hs_of_index);

void iterat_Pai_c_h(int *len_compact_atribut_i, int **index_of_hs_atr, double **Ex,
	int len_new_data, int y_dim, int *L_hs, double ****Pai_c_h, int **new_data,
	int L_all_h, double delta);

double Record_Convergence(double ****Pai_c_h_shadow,double ****Pai_c_h,int *len_compact_atribut_i,int y_dim,
				int *L_hs,double *Ph_shadow,int L_all_h,double *Ph);




void free_all(int **hs_of_index, double **Ex, double **temp_factor,
	int *len_i_class_factor, int **class_factor, int **len_ij_factor, int *len_i_factor,
	int ***factor, int *len_interval, int **interval, int len_new_data, int *data_repeat,
	int **new_data, int L_all_h, int *L_hs, int *len_h_atr_i, int **h_atr, int y_dim,
	double ****Pai_c_h_shadow,double *Ph_shadow,int *len_compact_atribut_i,int len_h,double **Ph_i);

