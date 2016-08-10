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
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	//���� �Ҵ� �κ�

static char *line = NULL;
static int max_line_len;

using namespace std;

struct svm_parameter par;		//�Ķ����
struct svm_model *model;		//��
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
		}

	}
}