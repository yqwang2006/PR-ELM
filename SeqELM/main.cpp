#include "POSELM.h"
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <sys/time.h>
//要求数据文件的格式为：样本数 * （样本维度+标签维度）
//第一列数据为标签，后面多列为样本，一行表示一个样本
//dat文件中按照matlab格式存放，即按列存放
//例如：matlab中生成的data大小为1000*（784+1),则生成的dat文件前1000个数据为data的第一列的数据
int main(int argc, char **argv){

	int NumberClasses;
	if(argc >= 2)
		NumberClasses = atoi(argv[1]);
	else
		NumberClasses = 5;
	
	timeval all_time_start, all_time_stop, combine_start, combine_stop, elm_start,elm_stop;
	double all_time, combine_time, timeuse, elm_time, test_time;
	all_time = 0;
	combine_time = 0;

	

	std::string filename = "data/Trainset.dat";
	mat TrainData, TrainLabels;
	read_dat_file(TrainData, TrainLabels, filename,DATADIM);
	


	std::string testfilename = "data/Testset.dat";
	mat TestData, TestLabels;
	read_dat_file(TestData, TestLabels, testfilename, DATADIM);
	
	gettimeofday(&all_time_start,0);
	int batch_number;
	int batch_size ;
	if(argc >= 3)
		batch_number = atoi(argv[2]);
	else
		batch_number = 1;
	
	double C = 1;
	int layers = 1;

	int inputDim = TrainData.n_rows;
	int NumberSamples = TrainData.n_cols;

	batch_size = NumberSamples / batch_number;

	cout << "InputDim = " << inputDim << endl;
	cout << "NumberSamples = " << NumberSamples << endl;

	
	
	mat *InputWeight = new mat[layers];
	mat *Bias = new mat[layers];
	std::string *ActFunc = new std::string[layers];
	int *hiddenDim = new int[layers+1];


	hiddenDim[0] = inputDim;

	if(argc >= 4)
		hiddenDim[1] = atoi(argv[3]);
	else
		hiddenDim[1] = 1000;

	for(int i = 0;i < layers; i++){
		InputWeight[i] = randu(hiddenDim[i+1],hiddenDim[i]) * 2 - 1;
		Bias[i] = randu(hiddenDim[i+1],1);
		ActFunc[i] = "sig";
	}

	//broadcast to each MPI node

	//下面的过程是在本届点的算法，每个节点都会执行该程序
	int start_loc = 0;
	int end_loc = 0;

	gettimeofday(&elm_start,0);

	mat *beta = new mat[batch_number];
	mat *HH = new mat[batch_number];
	for(int batch = 0; batch < batch_number; batch++){
		start_loc = batch * batch_size;
		end_loc = (batch+1) * batch_size - 1;
		if (end_loc >= NumberSamples)
			end_loc = NumberSamples-1;
		mat TrainData_batch = TrainData.cols(start_loc,end_loc);
		mat TrainLabels_batch = TrainLabels.rows(start_loc,end_loc);
		run_ELM(TrainData_batch,TrainLabels_batch,beta[batch],HH[batch],InputWeight,Bias, ActFunc,C,layers,NumberClasses);

		//double err_rate = ELM_test(TestData, TestLabels, InputWeight, beta[batch], Bias, ActFunc, C, layers, NumberClasses);

		//printf("err_rate = %f\n", err_rate);

	}
	gettimeofday(&elm_stop,0);
	timeuse = 1000000 * (elm_stop.tv_sec - elm_start.tv_sec) + (elm_stop.tv_usec - elm_start.tv_usec);
	elm_time = timeuse / 1000000;


	gettimeofday(&combine_start,0);
	

	mat OutputWeight =  beta[0];
	mat OutHH = HH[0];
	mat eye_C = (1.0/C)*eye(hiddenDim[layers],hiddenDim[layers]);
	mat combine_HT = (OutHH + eye_C) * OutputWeight;

	for(int batch = 1; batch < batch_number; batch++){

		combine_HT = combine_HT + (HH[batch] + eye_C) * beta[batch];
		
		OutHH = OutHH + HH[batch];
	}
	if(batch_number > 1){
		combine_model(OutputWeight,OutHH,combine_HT, C);
	}

	gettimeofday(&combine_stop,0);
	timeuse = 1000000 * (combine_stop.tv_sec - combine_start.tv_sec) + (combine_stop.tv_usec - combine_start.tv_usec);
	combine_time = timeuse / 1000000;


	gettimeofday(&combine_start,0);
	
	double err_rate = ELM_test(TestData, TestLabels, InputWeight, OutputWeight, Bias, ActFunc, C, layers, NumberClasses);

	gettimeofday(&combine_stop,0);
	timeuse = 1000000 * (combine_stop.tv_sec - combine_start.tv_sec) + (combine_stop.tv_usec - combine_start.tv_usec);
	test_time = timeuse / 1000000;

	printf("Testing accu = %f%%\n", (1-err_rate)*100);


	gettimeofday(&all_time_stop,0);
	timeuse = 1000000 * (all_time_stop.tv_sec - all_time_start.tv_sec) + (all_time_stop.tv_usec - all_time_start.tv_usec);
	all_time = timeuse / 1000000;


	printf("ELM costs %f s\n", elm_time);
	printf("Combine costs %f s\n", combine_time);
	printf("Test costs %f s\n", test_time);
	printf("The whole program costs %f s \n", all_time);

	delete []beta;
	delete []HH;
	delete []InputWeight;
	delete []Bias;
	delete []ActFunc;
	delete []hiddenDim;
	return 0;
}
