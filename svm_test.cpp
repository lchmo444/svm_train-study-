#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <string>
#include <new>
#include <thread>		//변경
#include <mutex>		//변경
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	//동적 할당 부분

static char *line = NULL;
static int max_line_len;

using namespace std;

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
}

void param_opt(char *input_file, char * output_file)
{
