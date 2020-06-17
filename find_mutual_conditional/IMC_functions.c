#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "IMC_functions.h"
double sum(double *array,int l)
{
	double s = 0;
	for(int i =0;i<l;i++)
	{
		s+=array[i];
	}
	return s;
}


void find_PAiAj(double ***PAiAj,int max_a,int max_b,int max_c,int *a,int *b,int *c,int X_dim)
{
	for (int class_=0;class_<max_c;class_++)
	{
		PAiAj[class_] = (double **)malloc(max_a*sizeof(double*));
		for (int i=0;i<max_a;i++)
		{
			PAiAj[class_][i] = (double *)malloc(max_b*sizeof(double));
			for (int j =0;j<max_b;j++)
			{
				PAiAj[class_][i][j] = 0;
			}
		}
	}
	
	int *sums = (int *)malloc(max_c*sizeof(int));
	for (int i=0;i<max_c;i++)
	{
		sums[i] = 0;
	}
	for (int i=0;i<X_dim;i++)
	{
		sums[c[i]] += 1;
		PAiAj[c[i]][a[i]][b[i]] += 1;
	}
	
	
	for (int class_=0;class_<max_c;class_++)
	{
		for (int i =0;i<max_a;i++)
		{
			for (int j =0;j<max_b;j++)
			{
				PAiAj[class_][i][j] /= (double)sums[class_];
			}
		}
	}
	free(sums);
}



double find_I(double ***PAiAj,double **PAi,double **PAj,int max_a,int max_b,int max_c,double *Pc)
{
	double I = 0;
	for (int class_=0;class_<max_c;class_++)
	{
		double a = 0;
		for (int i =0;i<max_a;i++)
		{
			for (int j =0;j<max_b;j++)
			{
				if (PAiAj[class_][i][j] !=0)
				{
					a += PAiAj[class_][i][j]*log2(PAiAj[class_][i][j]/PAi[class_][i]/PAj[class_][j]);
				}
			}
		}
		a*= Pc[class_];
		I += a;
	}
	return I;
}


void free_PAiAj(double ***PAiAj,int max_a,int max_b,int max_c)
{
	for (int class_=0;class_<max_c;class_++)
	{
		for (int i=0;i<max_a;i++)
		{
			free(PAiAj[class_][i]);
		}
		free(PAiAj[class_]);
	}
	free(PAiAj);
}


void find_max_attributes(int *max_attribute,int y_dim,int X_dim,int **X,int *max_y,int *y)
{
	for (int i =0 ;i<y_dim;i++)
	{
		max_attribute[i] = X[0][i];
		for (int j =1 ;j<X_dim;j++)
		{
			max_attribute[i] = (((max_attribute[i])>(X[j][i]))?(max_attribute[i]):(X[j][i]));
		}
		max_attribute[i]++;
	}
	(*max_y) = y[0];
	for (int j =1 ;j<X_dim;j++)
	{
		(*max_y) = ((((*max_y))>(y[j]))?((*max_y)):(y[j]));
	}
	(*max_y)++;
}



void find_PAi(double ***PAi,int X_dim,int y_dim,int max_y,int *max_attribute,int **X,int *y)
{
	for (int atr=0;atr<y_dim;atr++)
	{
		PAi[atr] = (double **)malloc(max_y*sizeof(double*));
		for (int class_=0;class_<max_y;class_++)
		{
			PAi[atr][class_] = (double *)malloc(max_attribute[atr]*sizeof(double));
			for (int i =0;i<max_attribute[atr];i++)
			{
				PAi[atr][class_][i] = 0;
			}
		}
	}
	
	for (int i=0;i<X_dim;i++)
	{
		for (int atr=0;atr<y_dim;atr++)
		{
			PAi[atr][y[i]][X[i][atr]]++;
		}
	}
	
	for (int atr=0;atr<y_dim;atr++)
	{
		for (int class_=0;class_<max_y;class_++)
		{
			double s = sum(PAi[atr][class_],max_attribute[atr]);
			for (int i =0;i<max_attribute[atr];i++)
			{
				PAi[atr][class_][i] /= s;
			}
		}
	}
	
}



void find_X_traspose(int **X_traspose,int **X,int X_dim,int y_dim)
{
	for (int i=0;i<y_dim;i++)
	{
		X_traspose[i] = (int *)malloc(X_dim*sizeof(int));
		for (int j=0;j<X_dim;j++)
		{
			X_traspose[i][j] = X[j][i];
		}
	}
}

void find_Pc(double *Pc,int X_dim,int y_dim,int *y,int max_y)
{
	for(int i =0;i<max_y; i++)
	{
		Pc[i] = 0;
	}
	for(int i =0;i<X_dim; i++)
	{
		Pc[y[i]]++;
	}
	for(int i =0;i<max_y; i++)
	{
		Pc[i] /= X_dim;
	}
}


void free_all(double *Pc,int **X_traspose,double ***PAi,int *max_attribute,int y_dim,int max_y)
{
	free(Pc);
	
	for (int i=0;i<y_dim;i++)
	{
		free(X_traspose[i]);
	}
	free(X_traspose);
	
	for (int atr=0;atr<y_dim;atr++)
	{
		for (int class_=0;class_<max_y;class_++)
		{
			free(PAi[atr][class_]);
		}
		free(PAi[atr]);
	}
	free(PAi);
	
	free(max_attribute);
}
