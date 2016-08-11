#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <string>
#include <new>
#include <thread>		//����
#include <mutex>		//����
#include "svm_test.h"
typedef float Qfloat;

#ifndef max
template <class T> static inline T max(T x, T y) { return (x>y) ? x : y; }
#endif

template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst, (void *)src, sizeof(T)*n);
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	//���� �Ҵ� �κ�


static char *line = NULL;
static int max_line_len;

using namespace std;

struct svm_parameter par;		//�Ķ����
struct svm_model *model;		//��
struct svm_problem prob;		//
struct svm_node *x_space;		//

class Cache
{
public:
	Cache(int l, long int size);
	~Cache();
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);		//swap

private:
	int l;
	long int size;
	
	struct head_t
	{
		head_t *prev, *next;
		Qfloat *data;
		int len;
	};
	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_, long int size_) :l(l_), size(size_)
{
	head = (head_t *)calloc(l, sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int)l);	// cache must be large enough for two columns	//original
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	head_t *h2 = &head[7];


	if (h->len) lru_delete(h);
	int more = len - h->len;		//ó������ �� ���� ���� �ε��� ���̸� �A��.

	if (more > 0)
	{
		// free old space
		//����� ����
		//#pragma omp parallel firstprivate()
		{
			//#pragma omp single
			while (size < more)	//original		//size�� ũ�� while�� ���� �ȵ�
				//for(more;  size < more;more)
			{
				head_t *old = lru_head.next;
				lru_delete(old);
				free(old->data);

				//#pragma omp task
				size += old->len;
				old->data = 0;
				old->len = 0;
			}
		}
		// allocate new space
		h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat)*len);

		h2->data = (Qfloat *)realloc(h2->data + 1, sizeof(Qfloat)*len);
		size -= more;
		swap(h->len, len);

	}

	lru_insert(h);
	*data = h->data;

	return len;
}

void Cache::swap_index(int i, int j)
{
	if (i == j) return;

	if (head[i].len) lru_delete(&head[i]);
	if (head[j].len) lru_delete(&head[j]);
	swap(head[i].data, head[j].data);
	swap(head[i].len, head[j].len);
	if (head[i].len) lru_insert(&head[i]);
	if (head[j].len) lru_insert(&head[j]);

	if (i>j) swap(i, j);
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
	{
		if (h->len > i)
		{
			if (h->len > j)
				swap(h->data[i], h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}









class Kernel{
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
		const svm_parameter& param);

protected:
	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
	}

};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& par)
	:kernel_type(par.kernel_type), degree(par.degree),
	gamma(par.gamma), coef0(par.coef0)
{
	if(kernel_type==3){
		kernel_function = &Kernel::kernel_rbf;
	}

	clone(x, x_, l);

	if (kernel_type == 3)
	{
		x_square = new double[l];
		for (int i = 0; i<l; i++)
			x_square[i] = dot(x[i], x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while (px->index != -1 && py->index != -1)
	{
		if (px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if (px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}


double Kernel::k_function(const svm_node *x, const svm_node *y,
	const svm_parameter& param)
{
	if(param.kernel_type ==3)
	{
		double sum = 0;
		while (x->index != -1 && y->index != -1)
		{
			if (x->index == y->index)
			{
				double d = x->value - y->value;
				sum += d*d;
				++x;
				++y;
			}
			else
			{
				if (x->index > y->index)
				{
					sum += y->value * y->value;
					++y;
				}
				else
				{
					sum += x->value * x->value;
					++x;
				}
			}
		}

		while (x->index != -1)
		{
			sum += x->value * x->value;
			++x;
		}

		while (y->index != -1)
		{
			sum += y->value * y->value;
			++y;
		}

		return exp(-param.gamma*sum);
	}
	
}



class SVC_Q : public Kernel
{
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& par, const char *y_) :Kernel(prob.l, prob.x, par)
	{
		clone(y, y_, prob.l);
		cache = new Cache(prob.l, (long int)(par.cache_size*(1 << 20)));		//cache ���úκ�
		QD = new double[prob.l];
		for (int i = 0; i<prob.l; i++)
			QD[i] = (this->*kernel_function)(i, i);
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;	//����,	//original null ����

		int start, j;
		int k_start;			//int	//����
		
		if ((start = cache->get_data(i, &data, len)) < len)		//data[j]�� ó���� ������ �� ����
		{
			k_start = start;		//����
			{
				
				for (j = start; j < len; j++)			//���� �߿� original		//data[j]�� �׽� �ٲ�
					data[j] = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));	//original
				
			}
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i, j);
		//original Kernel
		SVC_Q::swap_index(i, j);
		swap(y[i], y[j]);
		swap(QD[i], QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}

private:
	char *y;
	Cache *cache;
	double *QD;
};






class Solver{
public:
	double Gmax;			//ù��° Gmax
	double Gmax2;			//�ι�° Gmax
	int Gmin_idx;

	Solver(){};
	virtual ~Solver(){};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q)
};



















static char * read(FILE *in)
{
	int len;

	if (fgets(line, max_line_len, in) == NULL)
		return NULL;
	
	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, in) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv)
{
	const char *error_msg;

	string in_path = "\C:\\Users\\lee\\data.scale";
	string out_path = "\C:\\Users\\lee\\data_2.scale";


	char * input_file = new char[in_path.length() + 1];
	char * output_file = new char[out_path.length() + 1];
	strcpy(input_file, in_path.c_str());
	strcpy(output_file, out_path.c_str());

	param_opt(input_file, output_file);

	model = svm_train(&prob, &par);

}

void param_opt(char *input_file, char * output_file)
{
	/*default mode*/
	double weight_val[2] = { 0.87079200, 1 };
	par.svm_type = 1;	//C_SVC								////C_SVC �� 1
	par.kernel_type = 3;	//RBF							////RBF �� 3
	par.degree = 3;
	par.gamma = 0.002066115;
	//par.coef0 = 0;
	par.eps = 1;		//�ԽǷ� �κ�
	par.shrinking = 1;
	par.probability = 0;
	par.nr_weight = 0;
	par.weight_label = NULL;
	par.weight = NULL;

	par.nr_weight = 1;
	par.weight_label = (int *)realloc(par.weight_label, sizeof(int)*par.nr_weight);
	par.weight = (double *)realloc(par.weight, sizeof(double)*par.nr_weight);
	par.weight_label[0] = 1;
	par.weight[0] = weight_val[0];

	par.nr_weight = 2;
	par.weight_label = (int *)realloc(par.weight_label, sizeof(int)*par.nr_weight);
	par.weight = (double *)realloc(par.weight, sizeof(double)*par.nr_weight);
	par.weight_label[1] = -1;
	par.weight[1] = weight_val[1];

	par.cache_size = 900;
	par.C = 32;

}

void read_data(char *input_file)
{
	int max_index, instmax_index, i;	//�� ���� �ִ� feature index, index
	size_t elements, j;	//��ü ��ó ���� ù���� �͵� ����
	FILE *fp = fopen(input_file, "r");
	char *endptr;
	char * idx, *val, *label;

	if (fp == NULL)
	{
		exit(1);
	}

	prob.l = 0;		//��ü ���� �� �ʱ�ȭ
	elements = 0;	//��� �ʱ�ȭ

	max_line_len = 1024;	//���ٿ� �о���� �ִ� ����
	line = Malloc(char, max_line_len);
	while (read(fp) != NULL)
	{
		char *p = strtok(line, " \t");	//label

		//feature
		while (1)
		{
			p = strtok(NULL, "\t");		//strtok_s �� ���Ͽ� ����
			if (p == NULL || *p == '\n') //������ ��ó ���� "\n" break ������
				break;
			++elements;
		}
		++elements;	//������ ��ó ���� \n���� ��� �� �߰�
		++prob.l;	//���� �� ī��Ʈ
	}
	rewind(fp);		//���� ������ ��ġ ó������ �̵� ��Ų��.

	prob.y = Malloc(double, prob.l);	//�� �� �� 1�Ǵ� -1 ����
	prob.x = Malloc(struct svm_node *, prob.l);	//
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;		//max_index �� �ʱ�ȭ
	j = 0;
	for (i = 0; i < prob.l; i++)
	{
		instmax_index = -1;	 //precomputed kernel �ƴϸ� �ʱⰪ�� -1���ʹ�
		read(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label = NULL)	//empty line
			exit(1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit(1);

		while (1)
		{
			idx = strtok(NULL, ":"); //1: 0.43534 1�κ�
			val = strtok(NULL, " \t");

			if (val = NULL)
				break;

			errno = 0;
			////////char * ���� NULL, 0, '\0' �� Ÿ���� �ٸ���'\0' �� �ֽ��ϴ�/////////
			/////isspace ǥ�� ȭ��Ʈ �����̽� ' ', '\t', '\n', '\v', '\f', '\r'
			////x_space[j]�� �������(1, 0.44444), (2, 0.33333), (3. 0.64444), (5. 0.24444)�� �����
			x_space[j].index = (int)strtol(idx, &endptr, 10);	//10���� �� �Է�
			//if (endptr == idx || errno != 0 || (*endptr != !isspace(*endptr))
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= instmax_index)
				exit(1);
			else
				instmax_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit(1);

			++j;
		}

		if (instmax_index > max_index)
			max_index = instmax_index;
		x_space[j++].index = -1;		//������� �ε��� -1 ����
	}

	if (par.gamma == 0 && max_index > 0)		//// default�� 0���� ������ ���� 1/num_features, ���� �ִ밪 index
		par.gamma = 1.0 / max_index;		//���� 1/#feature element ���� ����

	fclose(fp);
}

svm_model *svm_train(const svm_problem * prob, const svm_parameter *par)
{
	svm_model * model = Malloc(svm_model, 1);
	model->param = *par;
	model->free_sv = 0;

	// classification
	int l = prob->l;
	int nr_class;
	int *label = NULL;		////Ŭ���� 2���� 1�ϋ� 1������ -1�̸� -1 �� ����
	int *start = NULL;		//+1�� ���� ��ġ ������� +1�� �������� 4�̸� 4 -1�� ������ ��ġ�� 8�̸� 8
	int *count = NULL;		////1�϶��� �� ����, -1�϶��� �Ѱ���
	int *perm = Malloc(int, l);

	// group training data of the same class
	svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
	
	svm_node **x = Malloc(svm_node *, l);
	
	int i;
	for (i = 0; i<l; i++)					/////////////////////////////////�߿�
		x[i] = prob->x[perm[i]];		//1:0.4444 2:0.333 3:0.432 4:0.2111 ������ -1 ���� �������

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for (i = 0; i<nr_class; i++)
		weighted_C[i] = par->C;			//0���� 32 1���� 32

	int j;
	for (i = 0; i < par->nr_weight;i++)		//Ŭ�������� weight�� ���� weight �� 2�� �׷��� nr_weight�� 2��
	{
		for (j = 0; j < nr_class; j++){
			weighted_C[j] *= par->weight[i];
		}
	}

	// train k*(k-1)/2 models

	bool *nonzero = Malloc(bool, l);
	for (i = 0; i<l; i++)
		nonzero[i] = false;				//������ false�� �ʱ�ȭ

	decision_function *f = Malloc(decision_function, nr_class*(nr_class - 1) / 2);		//rho, alpha �ʱ�ȭ

	//////nr_class �� 2�� �����ϴ�. �ΰ��� �з��Ҷ�
	int p = 0;
	for (i = 0; i < nr_class; i++)
		for (int j = i + 1; j < nr_class; j++)
		{
			svm_problem sub_prob;
			int si = start[i], sj = start[j];		//+1�� 4 -1�� 8(���ۺκ�)
			int ci = count[i], cj = count[j];		//+1�� ������ 4 -1�� ������ 4
			sub_prob.l = ci + cj;
			sub_prob.x = Malloc(svm_node *, sub_prob.l);
			sub_prob.y = Malloc(double, sub_prob.l);
			int k;
			for (k = 0; k<ci; k++)		/////2		???1
			{
				sub_prob.x[k] = x[si + k];
				sub_prob.y[k] = +1;
			}
			for (k = 0; k<cj; k++)		/////3		????-1
			{
				sub_prob.x[ci + k] = x[sj + k];
				sub_prob.y[ci + k] = -1;
			}

			f[p] = svm_train_one(&sub_prob, par, weighted_C[i], weighted_C[j]);		//�ݺ�Ƚ��



		}


}

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double, prob->l);		//alpha �Ҵ� 
	SolutionInfo si;
	solve_c_svc(prob, param, alpha, &si, Cp, Cn);

}

static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;		//prob->l ��ü ����  prob->y �� 1�Ǵ� -1 prob->x		index��1,2,3,4,6,7 value�� 0.37931,0.310345,0.327869,0.66667
	double *minus_ones = new double[l];		//munus_ones �Ҵ�
	char *y = new char[l];

	int i;

	for (i = 0; i < l; i++)
	{
		alpha[i] = 0;	//alpha 0���� �ʱ�ȭ
		minus_ones[i] = -1;		//-1���� �ʱ�ȭ
		if (prob->y[i] > 0) y[i] = +1; else y[i] = -1;	//prob->y[i]���� ���� ���� ���� �Ҵ� 
	}

	Solve
}






// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine		//Ŭ���� �з� �κ�
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 2;		//Ŭä�� �з� ������
	int nr_class = 0;			//�ʱ� Ŭ���� ��
	int *label = Malloc(int, max_nr_class);		////Ŭ���� 2���� 1�ϋ� 1������ -1�̸� -1 �� ����
	int *count = Malloc(int, max_nr_class);		//1�϶��� �� ����, -1�϶��� �Ѱ���
	int *data_label = Malloc(int, l);			//Ŭ������ 2���� 4�� 4���� 1 -1�̸� �� ������ ���̺� 1�� ���� ��ŭ 0 �׸��� �״��� ���̺� -1�� 1�� ����
	int i;

	for (i = 0; i < l; i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for (j = 0; j < nr_class; j++)
		{
			if (this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if (j == nr_class)
		{
			label[nr_class] = this_label;
			label[nr_class] = 1;
			++nr_class;
		}
	}


	int *start = Malloc(int, nr_class);		//2���� �з��������� nr_class�� 2�̴�
	start[0] = 0;

	for (i = 1; i<nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];
	
	/////perm�� ���ุŭ �ε��� ��ŭ �ʱ�ȭ�Ѵ�.
	for (i = 0; i<l; i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for (i = 1; i<nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];

	*nr_class_ret = nr_class;		//�� Ŭ���� ����
	*label_ret = label;		//Ŭ������ 2���϶� �Ѱ��� +1 �ϳ��� -1�� �� �ִ�
	*start_ret = start;		//+1�� ���� ��ġ ������� +1�� �������� 4�̸� 4 -1�� ������ ��ġ�� 8�̸� 8
	*count_ret = count;		//+1�� �� ����. -1�� �� ����
	free(data_label);
}