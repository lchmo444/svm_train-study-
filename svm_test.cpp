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
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	//동적 할당 부분

static char *line = NULL;
static int max_line_len;

using namespace std;

struct svm_parameter par;		//파라메터
struct svm_model *model;		//모델
struct svm_problem prob;		//
struct svm_node *x_space;		//

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

}

void param_opt(char *input_file, char * output_file)
{
	/*default mode*/
	double weight_val[2] = { 0.87079200, 1 };
	par.svm_type = 1;	//C_SVC
	par.kernel_type = 3;	//RBF
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
		}

	}
}