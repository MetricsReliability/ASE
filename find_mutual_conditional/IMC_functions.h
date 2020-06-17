

double sum(double *array,int l);

void find_PAiAj(double ***PAiAj,int max_a,int max_b,int max_c,int *a,int *b,int *c,int X_dim);

double find_I(double ***PAiAj,double **PAi,double **PAj,int max_a,int max_b,int max_c,double *Pc);

void free_PAiAj(double ***PAiAj,int max_a,int max_b,int max_c);


void find_max_attributes(int *max_attribute,int y_dim,int X_dim,int **X,int *max_y,int *y);


void find_PAi(double ***PAi,int X_dim,int y_dim,int max_y,int *max_attribute,int **X,int *y);



void find_X_traspose(int **X_traspose,int **X,int X_dim,int y_dim);

void find_Pc(double *Pc,int X_dim,int y_dim,int *y,int max_y);

void free_all(double *Pc,int **X_traspose,double ***PAi,int *max_attribute,int y_dim,int max_y);


