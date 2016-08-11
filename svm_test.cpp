#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <string>
#include <new>
#include <thread>		//변경
#include <mutex>		//변경
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

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	//동적 할당 부분


static char *line = NULL;
static int max_line_len;

using namespace std;

struct svm_parameter par;		//파라메터
struct svm_model *model;		//모델
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
	int more = len - h->len;		//처음에는 총 길이 에서 인덱스 길이를 뺸다.

	if (more > 0)
	{
		// free old space
		//사용자 변경
		//#pragma omp parallel firstprivate()
		{
			//#pragma omp single
			while (size < more)	//original		//size가 크면 while문 실행 안됨
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
		cache = new Cache(prob.l, (long int)(par.cache_size*(1 << 20)));		//cache 관련부분
		QD = new double[prob.l];
		for (int i = 0; i<prob.l; i++)
			QD[i] = (this->*kernel_function)(i, i);
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;	//변경,	//original null 없음

		int start, j;
		int k_start;			//int	//변경
		
		if ((start = cache->get_data(i, &data, len)) < len)		//data[j]는 처음값 고정일 수 있음
		{
			k_start = start;		//변경
			{
				
				for (j = start; j < len; j++)			//변경 중요 original		//data[j]는 항시 바뀜
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
	double Gmax;			//첫번째 Gmax
	double Gmax2;			//두번째 Gmax
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
	par.svm_type = 1;	//C_SVC								////C_SVC 는 1
	par.kernel_type = 3;	//RBF							////RBF 는 3
	par.degree = 3;
	par.gamma = 0.002066115;
	//par.coef0 = 0;
	par.eps = 1;		//입실론 부분
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
	int max_index, instmax_index, i;	//각 행의 최대 feature index, index
	size_t elements, j;	//전체 피처 수등 첫번쨰 것도 포함
	FILE *fp = fopen(input_file, "r");
	char *endptr;
	char * idx, *val, *label;

	if (fp == NULL)
	{
		exit(1);
	}

	prob.l = 0;		//전체 행의 수 초기화
	elements = 0;	//요소 초기화

	max_line_len = 1024;	//한줄에 읽어오는 최대 숫자
	line = Malloc(char, max_line_len);
	while (read(fp) != NULL)
	{
		char *p = strtok(line, " \t");	//label

		//feature
		while (1)
		{
			p = strtok(NULL, "\t");		//strtok_s 에 대하여 실행
			if (p == NULL || *p == '\n') //마지막 피처 이후 "\n" break 내려감
				break;
			++elements;
		}
		++elements;	//마지막 피처 개수 \n이후 요소 값 추가
		++prob.l;	//행의 수 카운트
	}
	rewind(fp);		//파일 포인터 위치 처음으로 이동 시킨다.

	prob.y = Malloc(double, prob.l);	//행 의 라벨 1또는 -1 저장
	prob.x = Malloc(struct svm_node *, prob.l);	//
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;		//max_index 값 초기화
	j = 0;
	for (i = 0; i < prob.l; i++)
	{
		instmax_index = -1;	 //precomputed kernel 아니면 초기값은 -1부터다
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
			idx = strtok(NULL, ":"); //1: 0.43534 1부분
			val = strtok(NULL, " \t");

			if (val = NULL)
				break;

			errno = 0;
			////////char * 끝에 NULL, 0, '\0' 은 타입이 다른것'\0' 가 있습니다/////////
			/////isspace 표준 화이트 스페이스 ' ', '\t', '\n', '\v', '\f', '\r'
			////x_space[j]에 예를들면(1, 0.44444), (2, 0.33333), (3. 0.64444), (5. 0.24444)가 저장됨
			x_space[j].index = (int)strtol(idx, &endptr, 10);	//10진수 값 입력
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
		x_space[j++].index = -1;		//행끌나고 인덱스 -1 저장
	}

	if (par.gamma == 0 && max_index > 0)		//// default는 0으로 설정되 있음 1/num_features, 행의 최대값 index
		par.gamma = 1.0 / max_index;		//보통 1/#feature element 감마 있음

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
	int *label = NULL;		////클래스 2개면 1일떄 1로지정 -1이면 -1 로 지정
	int *start = NULL;		//+1의 시작 위치 예를들면 +1의 마지막이 4이면 4 -1의 마지막 위치는 8이면 8
	int *count = NULL;		////1일때의 총 갯수, -1일때의 총갯수
	int *perm = Malloc(int, l);

	// group training data of the same class
	svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
	
	svm_node **x = Malloc(svm_node *, l);
	
	int i;
	for (i = 0; i<l; i++)					/////////////////////////////////중요
		x[i] = prob->x[perm[i]];		//1:0.4444 2:0.333 3:0.432 4:0.2111 마지막 -1 저장 각행따라

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for (i = 0; i<nr_class; i++)
		weighted_C[i] = par->C;			//0번은 32 1번은 32

	int j;
	for (i = 0; i < par->nr_weight;i++)		//클래스마다 weight가 있음 weight 는 2개 그래서 nr_weight는 2개
	{
		for (j = 0; j < nr_class; j++){
			weighted_C[j] *= par->weight[i];
		}
	}

	// train k*(k-1)/2 models

	bool *nonzero = Malloc(bool, l);
	for (i = 0; i<l; i++)
		nonzero[i] = false;				//각행을 false로 초기화

	decision_function *f = Malloc(decision_function, nr_class*(nr_class - 1) / 2);		//rho, alpha 초기화

	//////nr_class 는 2로 나뉩니다. 두개로 분류할때
	int p = 0;
	for (i = 0; i < nr_class; i++)
		for (int j = i + 1; j < nr_class; j++)
		{
			svm_problem sub_prob;
			int si = start[i], sj = start[j];		//+1는 4 -1는 8(시작부분)
			int ci = count[i], cj = count[j];		//+1은 갯수는 4 -1은 갯수는 4
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

			f[p] = svm_train_one(&sub_prob, par, weighted_C[i], weighted_C[j]);		//반복횟수



		}


}

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double, prob->l);		//alpha 할당 
	SolutionInfo si;
	solve_c_svc(prob, param, alpha, &si, Cp, Cn);

}

static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;		//prob->l 전체 길이  prob->y 는 1또는 -1 prob->x		index는1,2,3,4,6,7 value는 0.37931,0.310345,0.327869,0.66667
	double *minus_ones = new double[l];		//munus_ones 할당
	char *y = new char[l];

	int i;

	for (i = 0; i < l; i++)
	{
		alpha[i] = 0;	//alpha 0으로 초기화
		minus_ones[i] = -1;		//-1으로 초기화
		if (prob->y[i] > 0) y[i] = +1; else y[i] = -1;	//prob->y[i]값에 따라 값을 따로 할당 
	}

	Solve
}






// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine		//클래스 분류 부분
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 2;		//클채스 분류 가지수
	int nr_class = 0;			//초기 클래스 값
	int *label = Malloc(int, max_nr_class);		////클래스 2개면 1일떄 1로지정 -1이면 -1 로 지정
	int *count = Malloc(int, max_nr_class);		//1일때의 총 갯수, -1일때의 총갯수
	int *data_label = Malloc(int, l);			//클래스가 2개의 4개 4개가 1 -1이면 각 데이터 래이블 1의 갯수 만큼 0 그리고 그다음 래이블 -1은 1로 지정
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


	int *start = Malloc(int, nr_class);		//2개로 분류되있으면 nr_class는 2이다
	start[0] = 0;

	for (i = 1; i<nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];
	
	/////perm을 각행만큼 인덱스 만큼 초기화한다.
	for (i = 0; i<l; i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for (i = 1; i<nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];

	*nr_class_ret = nr_class;		//총 클래스 갯수
	*label_ret = label;		//클래스가 2개일때 한개는 +1 하나는 -1일 수 있다
	*start_ret = start;		//+1의 시작 위치 예를들면 +1의 마지막이 4이면 4 -1의 마지막 위치는 8이면 8
	*count_ret = count;		//+1의 총 갯수. -1의 총 갯수
	free(data_label);
}