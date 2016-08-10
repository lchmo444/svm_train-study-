
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
};

struct svm_model
{
	struct svm_parameter param;	/*parameter	*/
	int nr_class;				/*number of classes	*/
	int l;						/*total #SV	*/
	struct svm_node **SV;		/*Svs (SV[l])	*/
	double **sv_coef;			
	double *rho;
	double *probA;
	double *probB;
	int *sv_indices;

	int *label;
	int *nSV;
	int free_sv;
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