
struct svm_parameter
{
	int svm_type;			//1번 c_svc
	int kernel_type;		//3번 RBF
	int degree; /* for poly */
	double gamma;
	//double coef0;

	/*taining only*/
	double cache_size;	/* in MB*/
	double eps;			/* stopping criteria */
	double C;			/* for C_SVC	*/
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;
	int shrinking;
	int probability;		/*do probability estimates */
	double coef0;	/* for poly/sigmoid */

};

struct svm_model
{
	struct svm_parameter param;	/*parameter	*/
	int nr_class;				/*number of classes	*/
	int l;						/*total #SV	*/
	struct svm_node **SV;		/*Svs (SV[l])	*/
	double **sv_coef;			/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;				/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;				/* pariwise probability information */
	double *probB;
	int *sv_indices;			 /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	int *label;					/* label of each class (label[k]) */
	int *nSV;					/* number of SVs for each class (nSV[k]) */
	int free_sv;				/* 1 if svm_model is created by svm_load_model*/
};

struct svm_problem
{
	int l;			//행의 전체 길이
	double *y;
	struct svm_node **x;
};

struct svm_node
{
	int index;
	double value;
};

struct decision_function
{
	double *alpha;
	double rho;
};

struct SolutionInfo {
	double obj;
	double rho;
	double upper_bound_p;
	double upper_bound_n;
	double r;	// for Solver_NU
};