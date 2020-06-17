#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "processing_functions.h"



void find_h_atr(int **h_atr, int *len_h_atr_i, int y_dim, int ** h, int len_h, int* len_h_i)
{
	for (int i = 0; i < y_dim; i++)
	{
		int* binery = (int*)malloc(len_h * sizeof(int));
		for (int j = 0; j < len_h; j++)
		{
			binery[j] = 0;
			for (int k = 0; k < len_h_i[j]; k++)
			{
				if (i == h[j][k])
				{
					binery[j] = 1;
					break;
				}

			}
		}
		int tedad = 0;
		for (int j = 0; j < len_h; j++)
		{
			tedad += binery[j];
		}
		h_atr[i] = (int*)malloc(tedad * sizeof(int));
		len_h_atr_i[i] = tedad;
		int index = 0;
		for (int j = 0; j < len_h; j++)
		{
			if (binery[j] == 1)
			{
				h_atr[i][index] = j;
				index++;
			}
		}
		free(binery);
	}
}


void find_L_hs(int *L_hs, int y_dim, int *len_h_atr_i, int * L_h, int **h_atr)
{
	for (int i = 0; i < y_dim; i++)
	{
		L_hs[i] = 1;
		for (int j = 0; j < len_h_atr_i[i]; j++)
		{
			L_hs[i] *= L_h[h_atr[i][j]];
		}
	}
}

void find_Pc(double * Pc, int * y, int X_dim, int y_dim, int * len_compact_atribut_i)
{
	long long *c_repeat = (long long *)malloc(len_compact_atribut_i[y_dim] * sizeof(long long));
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		c_repeat[i] = 0;
	}
	for (int i = 0; i < X_dim; i++)
	{
		c_repeat[y[i]]++;
	}
	long long sum = 0;
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		sum += c_repeat[i];
	}
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		Pc[i] = ((double)c_repeat[i] / (double)sum);
		Pc[i] = log2(Pc[i]);
	}
	free(c_repeat);
}


int find_L_all_h(int *L_h, int len_h)
{
	int L_all_h = 1;
	for (int i = 0; i < len_h; i++)
	{
		L_all_h *= L_h[i];
	}
	return L_all_h;
}


void start_Ph(double * Ph, int L_all_h)
{
	for (int i = 0; i < L_all_h; i++)
	{
		Ph[i] = ((double)rand() / (RAND_MAX)) + 0.1;//((double)rand() / (RAND_MAX));
	}
	double sum = 0;
	for (int i = 0; i < L_all_h; i++)
	{
		sum += Ph[i];
	}
	for (int i = 0; i < L_all_h; i++)
	{
		Ph[i] /= sum;
		Ph[i] = log2(Ph[i]);
	}
}

void start_Ph_i(double **Ph_i, int len_h, int * L_h)
{
	for (int i = 0; i < len_h; i++)
	{
		Ph_i[i] = (double *)malloc(L_h[i] * sizeof(double));
		for (int j = 0; j < L_h[i]; j++)
		{
			Ph_i[i][j] = 0;
		}
	}
}

void start_Pai_c_h(double **** Pai_c_h, int * len_compact_atribut_i, int y_dim, int *L_hs)
{
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < L_hs[j]; k++)
			{
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					Pai_c_h[i][j][k][l] = ((double)rand() / (RAND_MAX))+0.1;//((double)rand() / (RAND_MAX))+1;
				}
				double sum = 0;
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					sum += Pai_c_h[i][j][k][l];
				}
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					Pai_c_h[i][j][k][l] /= sum;
					Pai_c_h[i][j][k][l] = log2(Pai_c_h[i][j][k][l]);
				}
			}
		}
	}
}

int determine_smaller(int * X1, int  y1, int * X2, int  y2, int y_dim)
{
	if (y1 > y2)
		return 2;
	else if (y1 < y2)
		return 1;
	for (int i = 0; i < y_dim; i++)
	{
		if (X1[i] > X2[i])
			return 2;
		else if (X1[i] < X2[i])
			return 1;
	}
	return 0;//equal!
}

void sotr_X(int ** X, int * y, int X_dim, int y_dim)
{
	if (X_dim < 2)
		return;
	int nesf = X_dim / 2;

	sotr_X(X, y, nesf, y_dim);
	sotr_X(X + nesf, y + nesf, X_dim - nesf, y_dim);

	int **tempX = (int **)malloc(X_dim * sizeof(int *));
	for (int i = 0; i < X_dim; i++)
	{
		tempX[i] = (int *)malloc(y_dim * sizeof(int));
	}
	int *tempy = (int *)malloc(X_dim * sizeof(int));

	int counter = 0;
	int counter1 = 0;
	int counter2 = nesf;
	while (1)
	{
		int a = determine_smaller(X[counter1], y[counter1], X[counter2], y[counter2], y_dim);
		if (a == 1)
		{
			for (int i = 0; i < y_dim; i++)
			{
				tempX[counter][i] = X[counter1][i];
			}
			tempy[counter] = y[counter1];
			counter++;
			counter1++;
		}
		else
		{
			for (int i = 0; i < y_dim; i++)
			{
				tempX[counter][i] = X[counter2][i];
			}
			tempy[counter] = y[counter2];
			counter++;
			counter2++;
		}
		if (counter1 == nesf)
		{
			for (counter2; counter2 < X_dim; counter2++)
			{
				for (int i = 0; i < y_dim; i++)
				{
					tempX[counter][i] = X[counter2][i];
				}
				tempy[counter] = y[counter2];
				counter++;
			}
			break;
		}
		else if (counter2 == X_dim)
		{
			for (counter1; counter1 < nesf; counter1++)
			{
				for (int i = 0; i < y_dim; i++)
				{
					tempX[counter][i] = X[counter1][i];
				}
				tempy[counter] = y[counter1];
				counter++;
			}
			break;
		}
	}

	for (int i = 0; i < X_dim; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			X[i][j] = tempX[i][j];
		}
		y[i] = tempy[i];
	}

	for (int i = 0; i < X_dim; i++)
	{
		free(tempX[i]);
	}
	free(tempX);
	free(tempy);

}

void find_new_data(int ***new_data, int **data_repeat, int *len_new_data,
	int ** X, int * y, int X_dim, int y_dim)
{
	*len_new_data = 1;
	int *binery = (int *)malloc(X_dim * sizeof(int));
	binery[0] = 1;
	for (int i = 1; i < X_dim; i++)
	{
		binery[i] = 0;
		if (determine_smaller(X[i - 1], y[i - 1], X[i], y[i], y_dim) != 0)
		{
			binery[i] = 1;
			(*len_new_data)++;
		}
	}
	(*new_data) = (int **)malloc((*len_new_data) * sizeof(int *));
	(*data_repeat) = (int *)malloc((*len_new_data) * sizeof(int));
	for (int i = 0; i < (*len_new_data); i++)
	{
		(*data_repeat)[i] = 0;
	}
	int index = 0;
	for (int i = 0; i < X_dim; i++)
	{
		if (binery[i] == 1)
		{
			(*new_data)[index] = (int *)malloc((y_dim + 1) * sizeof(int));
			for (int j = 0; j < (y_dim); j++)
			{
				(*new_data)[index][j] = X[i][j];
			}
			(*new_data)[index][y_dim] = y[i];
			index++;
		}
		(*data_repeat)[index - 1]++;
	}
	free(binery);
}


int interval_len(int **new_data, int len_new_data, int feature, int start, int end)
{
	if (start >= end)
		return 0;
	int len_interval = 1;
	for (int i = start + 1; i < end; i++)
	{
		if (new_data[i][feature] != new_data[i - 1][feature])
		{
			len_interval++;
		}
	}
	return len_interval;
}
void verify_interval(int **new_data, int len_new_data, int feature, int start, int end, int *interval)
{
	if (start >= end)
		return;
	interval[0] = start;
	int interval_idx = 0;
	for (int i = start + 1; i < end; i++)
	{
		if (new_data[i][feature] != new_data[i - 1][feature])
		{
			interval_idx++;
			interval[interval_idx] = i;
		}
	}
}

void find_interval(int **interval, int *len_interval, int **new_data, int len_new_data, int y_dim)
{
	len_interval[0] = interval_len(new_data, len_new_data, y_dim, 0, len_new_data);
	interval[0] = (int *)malloc(len_interval[0] * sizeof(int));
	verify_interval(new_data, len_new_data, y_dim, 0, len_new_data, interval[0]);

	for (int feature = 0; feature < y_dim; feature++)
	{
		len_interval[feature + 1] = 0;
		int *length = (int *)malloc(len_interval[feature] * sizeof(int));
		for (int i = 0; i < len_interval[feature] - 1; i++)
		{
			length[i] = len_interval[feature + 1];
			len_interval[feature + 1] +=
				interval_len(new_data, len_new_data, feature,
					interval[feature][i], interval[feature][i + 1]);
		}
		length[len_interval[feature] - 1] = len_interval[feature + 1];
		len_interval[feature + 1] +=
			interval_len(new_data, len_new_data, feature,
				interval[feature][len_interval[feature] - 1], len_new_data);
		interval[feature + 1] = (int *)malloc(len_interval[feature + 1] * sizeof(int));

		for (int i = 0; i < len_interval[feature] - 1; i++)
		{
			verify_interval(new_data, len_new_data, feature, interval[feature][i], interval[feature][i + 1],
				interval[feature + 1] + length[i]);
		}
		verify_interval(new_data, len_new_data, feature, interval[feature][len_interval[feature] - 1],
			len_new_data, interval[feature + 1] + length[len_interval[feature] - 1]);
		free(length);
	}
}


void find_factor(int ***factor, int *len_i_factor, int **len_ij_factor, int **class_factor,
	int *len_i_class_factor, int y_dim, int **new_data, int len_new_data,
	int **interval, int *len_interval)
{
	for (int feature = 0; feature < y_dim; feature++)
	{
		factor[feature] = (int **)malloc(len_interval[feature] * sizeof(int *));
		len_i_factor[feature] = len_interval[feature];
		len_ij_factor[feature] = (int *)malloc(len_interval[feature] * sizeof(int));

		class_factor[feature] = (int *)malloc(len_interval[feature] * sizeof(int));
		len_i_class_factor[feature] = len_interval[feature];
		for (int i = 0; i < len_interval[feature]; i++)
		{
			class_factor[feature][i] = new_data[interval[feature][i]][y_dim];
		}

		for (int i = 0; i < len_interval[feature] - 1; i++)
		{
			int length = 1;
			for (int j = interval[feature][i] + 1; j < interval[feature][i + 1]; j++)
			{
				if (new_data[j - 1][feature] != new_data[j][feature])
					length++;
			}
			factor[feature][i] = (int *)malloc(length * sizeof(int));
			len_ij_factor[feature][i] = length;

			factor[feature][i][0] = new_data[interval[feature][i]][feature];
			int index = 1;
			for (int j = interval[feature][i] + 1; j < interval[feature][i + 1]; j++)
			{
				if (new_data[j - 1][feature] != new_data[j][feature])
				{
					factor[feature][i][index] = new_data[j][feature];
					index++;
				}
			}
		}
		int length = 1;
		for (int j = interval[feature][len_interval[feature] - 1] + 1; j < len_new_data; j++)
		{
			if (new_data[j - 1][feature] != new_data[j][feature])
				length++;
		}
		factor[feature][len_interval[feature] - 1] = (int *)malloc(length * sizeof(int));
		len_ij_factor[feature][len_interval[feature] - 1] = length;

		factor[feature][len_interval[feature] - 1][0] = new_data[interval[feature][len_interval[feature] - 1]][feature];
		int index = 1;
		for (int j = interval[feature][len_interval[feature] - 1] + 1; j < len_new_data; j++)
		{
			if (new_data[j - 1][feature] != new_data[j][feature])
			{
				factor[feature][len_interval[feature] - 1][index] = new_data[j][feature];
				index++;
			}
		}
	}
}


double **build_temp_factor(int y_dim, int *len_interval)
{
	double **temp_factor = (double **)malloc(y_dim * sizeof(double *));
	for (int i = 0; i < y_dim; i++)
	{
		temp_factor[i] = (double *)malloc(len_interval[i + 1] * sizeof(double));
	}
	return temp_factor;
}



double **build_Ex(int len_new_data, int L_all_h)
{
	double **Ex = (double **)malloc(len_new_data * sizeof(double *));
	for (int i = 0; i < len_new_data; i++)
	{
		Ex[i] = (double *)malloc(L_all_h * sizeof(double));
	}
	return Ex;
}


int *find_h_from_index(int index, int *L_h, int len_h)
{
	int * h = (int *)malloc(len_h * sizeof(int));
	int a;
	for (int i = len_h - 1; i > -1; i--)
	{
		a = index % L_h[i];
		h[i] = a;
		index -= a;
		index /= L_h[i];
	}
	return h;
}


int **build_hs_of_index(int L_all_h, int *L_h, int len_h)
{
	int **hs_of_index = (int **)malloc(L_all_h * sizeof(int *));
	for (int i = 0; i < L_all_h; i++)
	{
		hs_of_index[i] = find_h_from_index(i, L_h, len_h);
	}
	return hs_of_index;
}

int find_index_h(int *h, int *L_h, int len_h)
{
	int index = 0;
	for (int i = 0; i < len_h - 1; i++)
	{
		index += h[i];
		index *= L_h[i + 1];
	}
	if (len_h != 0)
	{
		index += h[len_h - 1];
	}
	return index;
}

void build_index_of_hs_atr(int **index_of_hs_atr, int y_dim, int **h_atr, int *len_h_atr_i
	, int **hs_of_index, int L_all_h, int *L_h)
{
	for (int i = 0; i < y_dim; i++)
	{
		for (int j = 0; j < L_all_h; j++)
		{
			int *h_j = (int *)malloc(len_h_atr_i[i] * sizeof(int));
			int *L_h_j = (int *)malloc(len_h_atr_i[i] * sizeof(int));
			for (int k = 0; k < len_h_atr_i[i]; k++)
			{
				L_h_j[k] = L_h[h_atr[i][k]];
				h_j[k] = hs_of_index[j][h_atr[i][k]];
			}
			index_of_hs_atr[i][j] = find_index_h(h_j, L_h_j, len_h_atr_i[i]);
			free(h_j);
			free(L_h_j);
		}
	}
}


double find_max(double *array, int length)
{
	double maximum = array[0];
	for (int i = 1; i < length; i++)
	{
		maximum = (((maximum)>(array[i]))?(maximum):(array[i]));
	}
	return maximum;
}


void build_shadows(double *****Pai_c_h_shadow,double ****Pai_c_h,int *len_compact_atribut_i,int y_dim,
				int *L_hs,double **Ph_shadow,int L_all_h,double *Ph)
{
	(*Pai_c_h_shadow) = (double ****)malloc(len_compact_atribut_i[y_dim] * sizeof(double ***));
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		(*Pai_c_h_shadow)[i] = (double ***)malloc(y_dim * sizeof(double **));
		for (int j = 0; j < y_dim; j++)
		{
			(*Pai_c_h_shadow)[i][j] = (double **)malloc(L_hs[j] * sizeof(double *));
			for (int k = 0; k < L_hs[j]; k++)
			{
				(*Pai_c_h_shadow)[i][j][k] = (double *)malloc(len_compact_atribut_i[j] * sizeof(double));
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					(*Pai_c_h_shadow)[i][j][k][l] = Pai_c_h[i][j][k][l];
				}
			}
		}
	}
	(*Ph_shadow) = (double *)malloc(L_all_h * sizeof(double));
	for (int i = 0; i < L_all_h; i++)
	{
		(*Ph_shadow)[i] = Ph[i];
	}
	
}


void iterat_Ex(int L_all_h, int *len_i_factor, int **class_factor, int **len_ij_factor,
	double **temp_factor, double ****Pai_c_h, int **index_of_hs_atr, int ***factor,
	int y_dim, double *Ph, double **Ex, int *data_repeat, int len_new_data,double alpha)
{
	for (int j = 0; j < L_all_h; j++)
	{
		int tem_idx = 0;
		for (int k = 0; k < len_i_factor[0]; k++)
		{
			int class_ = class_factor[0][k];
			for (int l = 0; l < len_ij_factor[0][k]; l++)
			{
				temp_factor[0][tem_idx] = Pai_c_h[class_][0][index_of_hs_atr[0][j]][factor[0][k][l]] + Ph[j];
				tem_idx++;
			}
		}

		for (int i = 1; i < y_dim; i++)
		{
			tem_idx = 0;
			for (int k = 0; k < len_i_factor[i]; k++)
			{
				int class_ = class_factor[i][k];
				for (int l = 0; l < len_ij_factor[i][k]; l++)
				{
					temp_factor[i][tem_idx] = temp_factor[i - 1][k] + Pai_c_h[class_][i][index_of_hs_atr[i][j]][factor[i][k][l]];
					tem_idx++;
				}
			}
		}

		int i = y_dim - 1;
		for (int l = 0; l < len_new_data; l++)
		{
			Ex[l][j] = temp_factor[i][l];

		}
	}

	for (int i = 0; i < len_new_data; i++)
	{
		int temp = find_max(Ex[i], L_all_h);
		for (int k = 0; k < L_all_h; k++)
		{
			Ex[i][k] = exp2(Ex[i][k] - temp) + alpha;
		}
		double sum = 0;
		for (int k = 0; k < L_all_h; k++)
		{
			sum += Ex[i][k];
		}
		for (int k = 0; k < L_all_h; k++)
		{
			Ex[i][k] /= sum;
			Ex[i][k] *= data_repeat[i];
		}
	}
}



void iterat_Ph(int L_all_h, int **index_of_hs_atr, double *Ph, double **Ex,
	int len_new_data, int len_h, int mode_h, double **Ph_i, int *L_h, int **hs_of_index)
{
	
	if (mode_h == 1)
	{
		for (int i = 0; i < L_all_h; i++)
		{
			Ph[i] = 0;
			for (int k = 0; k < len_new_data; k++)
			{
				Ph[i] += Ex[k][i];
			}
		}

		for (int i = 0; i < len_h; i++)
		{
			for (int j = 0; j < L_h[i]; j++)
			{
				Ph_i[i][j] = 0;
			}
		}

		for (int i = 0; i < L_all_h; i++)
		{
			for (int j = 0; j < len_h; j++)
			{
				Ph_i[j][hs_of_index[i][j]] += Ph[i];
			}
		}

		for (int i = 0; i < len_h; i++)
		{
			double sum = 0;
			for (int j = 0; j < L_h[i]; j++)
			{
				sum += Ph_i[i][j];
			}
			for (int j = 0; j < L_h[i]; j++)
			{
				Ph_i[i][j] /= sum;
				Ph_i[i][j] = log2(Ph_i[i][j]);
			}
		}
		for (int i = 0; i < L_all_h; i++)
		{
			Ph[i] = 0;
			for (int j = 0; j < len_h; j++)
			{
				Ph[i] += Ph_i[j][hs_of_index[i][j]];
			}
		}

	}
	else
	{
		for (int i = 0; i < L_all_h; i++)
		{
			Ph[i] = 0;
			for (int k = 0; k < len_new_data; k++)
			{
				Ph[i] += Ex[k][i];
			}
		}
		double sum = 0;
		for (int i = 0; i < L_all_h; i++)
		{
			sum += Ph[i];
		}
		for (int i = 0; i < L_all_h; i++)
		{
			Ph[i] /= sum;
			Ph[i] = log2(Ph[i]);
		}
	}
	
}


void iterat_Pai_c_h(int *len_compact_atribut_i, int **index_of_hs_atr, double **Ex,
	int len_new_data, int y_dim, int *L_hs, double ****Pai_c_h, int **new_data,
	int L_all_h, double delta)
{
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < L_hs[j]; k++)
			{
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					Pai_c_h[i][j][k][l] = 0;
				}
			}
		}
	}

	for (int m = 0; m < len_new_data; m++)
	{
		int class_ = new_data[m][y_dim];
		for (int k = 0; k < y_dim; k++)
		{
			int l = new_data[m][k];
			for (int j = 0; j < L_all_h; j++)
			{
				//Pai_c_h[C=c][i][H=h][k] = P(Ai = ak | C=c,H=h)
				Pai_c_h[class_][k][index_of_hs_atr[k][j]][l] += Ex[m][j];
			}
		}
	}

	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < L_hs[j]; k++)
			{
				double sum = 0;
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					sum += Pai_c_h[i][j][k][l];
				}
				sum += delta * len_compact_atribut_i[j];
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					Pai_c_h[i][j][k][l] += delta;
					Pai_c_h[i][j][k][l] /= sum;
					Pai_c_h[i][j][k][l] = log2(Pai_c_h[i][j][k][l]);
				}
			}
		}
	}
}



double Record_Convergence(double ****Pai_c_h_shadow,double ****Pai_c_h,int *len_compact_atribut_i,int y_dim,
				int *L_hs,double *Ph_shadow,int L_all_h,double *Ph)
{
	double Convergence = 0;
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < L_hs[j]; k++)
			{
				for (int l = 0; l < len_compact_atribut_i[j]; l++)
				{
					Convergence += fabs(exp2(Pai_c_h_shadow[i][j][k][l]) - exp2(Pai_c_h[i][j][k][l]));
					Pai_c_h_shadow[i][j][k][l] = Pai_c_h[i][j][k][l];
				}
			}
		}
	}
	for (int i = 0; i < L_all_h; i++)
	{
		Convergence += fabs(exp2(Ph_shadow[i]) - exp2(Ph[i]));
		Ph_shadow[i] = Ph[i];
	}
	return Convergence;
}


void free_all(int **hs_of_index, double **Ex, double **temp_factor,
	int *len_i_class_factor, int **class_factor, int **len_ij_factor, int *len_i_factor,
	int ***factor, int *len_interval, int **interval, int len_new_data, int *data_repeat,
	int **new_data, int L_all_h, int *L_hs, int *len_h_atr_i, int **h_atr, int y_dim,
	double ****Pai_c_h_shadow,double *Ph_shadow,int *len_compact_atribut_i,int len_h,double **Ph_i)
{
	for (int i = 0; i < len_compact_atribut_i[y_dim]; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < L_hs[j]; k++)
			{
				free(Pai_c_h_shadow[i][j][k]);
			}
			free(Pai_c_h_shadow[i][j]);
		}
		free(Pai_c_h_shadow[i]);
	}
	free(Pai_c_h_shadow);
	free(Ph_shadow);
	
	for (int i = 0; i < L_all_h; i++)
	{
		free(hs_of_index[i]);
	}
	free(hs_of_index);


	for (int i = 0; i < len_new_data; i++)
	{
		free(Ex[i]);
	}
	free(Ex);


	for (int i = 0; i < y_dim; i++)
	{
		free(temp_factor[i]);
	}
	free(temp_factor);


	for (int i = 0; i < y_dim; i++)
	{
		for (int j = 0; j < len_i_factor[i]; j++)
		{
			free(factor[i][j]);
		}
		free(factor[i]);
		free(len_ij_factor[i]);
		free(class_factor[i]);
	}
	free(factor);
	free(len_ij_factor);
	free(class_factor);
	free(len_i_factor);
	free(len_i_class_factor);


	for (int i = 0; i < (y_dim + 1); i++)
	{
		free(interval[i]);
	}
	free(interval);
	free(len_interval);


	for (int i = 0; i < len_new_data; i++)
	{
		free(new_data[i]);
	}

	free(data_repeat);

	for (int i = 0; i < len_h; i++)
	{
		free(Ph_i[i]);
	}
	free(Ph_i);
	free(L_hs);

	for (int i = 0; i < y_dim; i++)
	{
		free(h_atr[i]);
	}

	free(h_atr);

	free(len_h_atr_i);
}

