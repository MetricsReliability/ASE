#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "IMC.h"
#include "IMC_functions.h"

DLLIMPORT double information_mutual_conditional(int *a,int *b,int *c,int X_dim,int max_a,
					int max_b,int max_c,double *Pc,double **PAi,double **PAj)
{
	
	double ***PAiAj = (double ***)malloc(max_c*sizeof(double**));
	find_PAiAj(PAiAj,max_a,max_b,max_c,a,b,c,X_dim);
	
	double I = find_I(PAiAj,PAi,PAj,max_a,max_b,max_c,Pc);
	
	free_PAiAj(PAiAj,max_a,max_b,max_c);
	
	return I;
}



DLLIMPORT void information_mutual_conditional_all(int **X,int *y,int X_dim,int y_dim,double **I)
{
	int *max_attribute = (int *)malloc(y_dim*sizeof(int));
	int max_y;
	find_max_attributes(max_attribute,y_dim,X_dim,X,&max_y,y);
	
	
	double ***PAi = (double ***)malloc(y_dim*sizeof(double**));
	find_PAi(PAi,X_dim,y_dim,max_y,max_attribute,X,y);
	
	
	int **X_traspose = (int **)malloc(y_dim*sizeof(int *));
	find_X_traspose(X_traspose,X,X_dim,y_dim);
	
	
	double *Pc = (double *)malloc(max_y*sizeof(double));
	find_Pc(Pc,X_dim,y_dim,y,max_y);
	
	for (int i =0;i<y_dim; i++)
	{
		for (int j =i;j<y_dim; j++)
		{
			I[i][j] = information_mutual_conditional(X_traspose[i],X_traspose[j],y,X_dim,max_attribute[i],
					max_attribute[j],max_y,Pc,PAi[i],PAi[j]);
		}
	}
	for (int i =0;i<y_dim; i++)
	{
		for (int j =0;j<i; j++)
		{
			I[i][j] = I[j][i];
		}
	}
	
	free_all(Pc,X_traspose,PAi,max_attribute,y_dim,max_y);
	
	
}
