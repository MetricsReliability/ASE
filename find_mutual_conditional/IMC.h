#ifndef _DLL_H_
#define _DLL_H_

#if BUILDING_DLL
#define DLLIMPORT __declspec(dllexport)
#else
#define DLLIMPORT __declspec(dllimport)
#endif

DLLIMPORT double information_mutual_conditional(int *a,int *b,int *c,int X_dim,int max_a,
					int max_b,int max_c,double *Pc,double **PAi,double **PAj);


DLLIMPORT void information_mutual_conditional_all(int **X,int *y,int X_dim,int y_dim,double **I);

#endif
