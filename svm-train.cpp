﻿#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <string>
#include <new>
#include <thread>		//변경
#include <mutex>		//변경

#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using namespace std;

int a = 0;



void print_null(const char *s) {}

void thread_model(char *, char, svm_model *);		//변경
std::mutex mtx;		//변경

void exit_with_help()
{
	printf(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(char *filename);		//
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;		//첫번째 model
struct svm_model *model_2;		//첫번째 model_2
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv)
{

	//char input_file_name[1024];		///뺀 부분
	//char model_file_name[1024];
	const char *error_msg;
	                                                                                             

	///////////////////////////////////////////////////////////
	/////사용자 임의로 넣어 준 부분 // input 및 output 파일 들
	string in_f = "\C:\\Users\\lee\\Desktop\\lee_chung_keun\\SVM material\\svm-train\\ConsoleApplication1\\Debug\\train_1200000_minimal_mini.scale";
	/*사용자 임의 폴더_1*/
	//string in_f = "\C:\\lee\\train_1200000_minimal.scale";


	char * input_file_name = new char[in_f.length() + 1];
	strcpy(input_file_name, in_f.c_str());

	string model_f = "\C:\\Users\\lee\\Desktop\\lee_chung_keun\\SVM material\\svm-train\\ConsoleApplication1\\Debug\\train_testver.model";

	string model_f2 = "\C:\\Users\\lee\\Desktop\\lee_chung_keun\\SVM material\\svm-train\\ConsoleApplication1\\Debug\\train_2.model";		
	
	/*사용자 임의 폴더_2*/
	//string model_f = "\C:\\lee\\train.model";
	//string model_f2 = "\C:\\lee\\train_2.model";


	char * model_file_name = new char[model_f.length() + 1];
	char * model_file_name_2 = new char[model_f2.length() + 1];
	strcpy(model_file_name, model_f.c_str());
	strcpy(model_file_name_2, model_f2.c_str());
	/////////////////////////////////////
	//static char *argv[7] = {};
	/*
	if (a < 1)
	{	
		a = a + 4;
		argc = 7;

		argv[0] = "";
		argv[3] = "-m 300";
		argv[4] = "-c 32";
		argv[1] = "-w1 0.00781245";
		argv[2] = "-w-1 1";
		argv[5] = (input_file_name);
		argv[6] = (model_file_name);

		main(argc, argv);
	}*/

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob, &param);

	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}

	if (cross_validation)
	{
		do_cross_validation();
	}
	else
	{


		std::thread thread1(thread_model, model_file_name, 'a',model);
		//std::thread thread2(thread_model, model_file_name_2, 'b', model_2);

		//thread1.joinable();
		//thread2.joinable();

		thread1.join();
		//thread2.join();
		//thread1.detach();
		//thread2.detach();
		//model = (svm_model*)(thread1.join());

	//	model = svm_train(&prob, &param);
		//mtx.unlock();

		//svm_save_model(model_file_name, model);
		//svm_destroy_param(&param);

	//	svm_free_and_destroy_model(&model);



		//svm_save_model(model_file_name + '1', model);		//original

		/*
		if (svm_save_model(model_file_name+ '2', model_2))		//model 저장 부분	//original
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}*/
		//svm_free_and_destroy_model(&model);
		//svm_free_and_destroy_model(&model_2);
	}
	/*
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	*/
	

	free(line);
	return 0;
}


void thread_model(char *model_file_name, char a, svm_model *bbb)			//thread model 함수부분
{
	//mtx.lock();
	bbb = svm_train(&prob, &param);
	//mtx.unlock();

	svm_save_model(model_file_name , bbb);
	//svm_destroy_param(&param);

	svm_free_and_destroy_model(&bbb);
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	//free(line);
	
}



void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	svm_cross_validation(&prob, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR ||
		param.svm_type == NU_SVR)
	{
		for (i = 0; i<prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y)*(v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy - sumv*sumy)*(prob.l*sumvy - sumv*sumy)) /
			((prob.l*sumvv - sumv*sumv)*(prob.l*sumyy - sumy*sumy))
			);
	}
	else
	{
		for (i = 0; i<prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.l);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void(*print_func)(const char*) = NULL;	// default printing to stdout



	double aa[2] = { 0.87079200, 1 };
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	//param.gamma = 0;	// 1/num_features	//original
	param.gamma = 0.002066115;
	param.coef0 = 0;
	param.nu = 0.5;
	//param.cache_size = 100;
	//param.C = 1;		//수정부분
	//param.eps = 1e-3;	//original
	param.eps = 1;	//사용자 부분
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	//변경 부분
	param.nr_weight = 1;
	param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
	param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
	param.weight_label[0] = 1;
	param.weight[0] = aa[0];


	param.nr_weight = 2;
	param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
	param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
	param.weight_label[1] = -1;
	param.weight[1] = aa[1];

	param.cache_size = 900;
	param.C = 32;


	cross_validation = 0;

	// parse options
	/*
	for (i = 1; i<argc; i++)
	{
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			exit_with_help();
		switch (argv[i - 1][1])
		{
		case 's':
			param.svm_type = atoi(argv[i]);
			break;
		case 't':
			param.kernel_type = atoi(argv[i]);
			break;
		case 'd':
			param.degree = atoi(argv[i]);
			break;
		case 'g':
			param.gamma = atof(argv[i - 1] + 3);
			break;
		case 'r':
			param.coef0 = atof(argv[i]);
			break;
		case 'n':
			param.nu = atof(argv[i]);
			break;
		case 'm':
			param.cache_size = atof(argv[i-1]+3);		//20160712 수정부분
			//param.cache_size = 300;
			break;
		case 'c':
			param.C = atof(argv[i - 1] + 3);
			//param.C = 32;
			break;
		case 'e':
			param.eps = atof(argv[i]);
			break;
		case 'p':
			param.p = atof(argv[i]);
			break;
		case 'h':
			param.shrinking = atoi(argv[i]);
			break;
		case 'b':
			param.probability = atoi(argv[i]);
			break;
		case 'q':
			print_func = &print_null;
			i--;
			break;
		case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2)
			{
				fprintf(stderr, "n-fold cross validation: n must >= 2\n");
				exit_with_help();
			}
			break;
		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;
		default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			exit_with_help();
		}
	}*/

	svm_set_print_string_function(print_func);

	// determine filenames
	
	/*
	if (i >= argc)
		exit_with_help();
		*/
	//strcpy(input_file_name, argv[i]);

	/*
	if (i<argc - 1)
		strcpy(model_file_name, argv[i + 1]);
	else
	{
		char *p = strrchr(argv[i], '/');
		if (p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name, "%s.model", p);
	}*/
}

// read in a problem (in svmlight format)

void read_problem(char *filename)	
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	//FILE *fp;			//변경
	FILE *fp = fopen(filename, "r");		//original
	//fopen_s(&fp, filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label

		// features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;
	j = 0;
	for (i = 0; i<prob.l; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(i + 1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0; i<prob.l; i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

