#include "POSELM.h"
#include "mpi.h"
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#define MAX_HID_NODE 9000*6
using namespace std;
//要求数据文件的格式为：样本数 * （样本维度+标签维度）
//第一列数据为标签，后面多列为样本，一行表示一个样本
//dat文件中按照matlab格式存放，即按列存放
//例如：matlab中生成的data大小为1000*（784+1),则生成的dat文件前1000个数据为data的第一列的数据
int main(int argc, char **argv){



	int myid,numprocs;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	int comm_tag = 1;

	int NumberClasses = 7;


	timeval all_time_start, all_time_stop, comm_time_start, comm_time_stop, combine_start, combine_stop,elm_start,elm_stop;
	double all_time, bcast_time, comm_time, combine_time, timeuse,elm_time,test_time;
	all_time = 0;
	bcast_time = 0;
	comm_time = 0;
	combine_time = 0;
	test_time = 0;


	
	
	string filename = "data/Trainset.dat";
	mat TrainData, TrainLabels;

	stringstream myid_str;
	myid_str << myid;
	//filename = filename + myid_str.str() + ".dat";
	
	cout << filename << endl;

	read_dat_file(TrainData, TrainLabels,  filename.c_str(), DATADIM);
	//read_dat_file(TrainData, TrainLabels,  "data/MnistTrainSet.dat", DATADIM);

	std::string testfilename = "data/Testset.dat";
	mat TestData, TestLabels;
	read_dat_file(TestData, TestLabels, testfilename.c_str(), DATADIM);

	gettimeofday(&all_time_start,0);

	double C = 1;
	int layers = 1;

	int inputDim = TrainData.n_rows;
	int NumberSamples = TrainData.n_cols;

	cout << "InputDim = " << inputDim << endl;
	cout << "NumberSamples = " << NumberSamples << endl;

	
	mat *InputWeight = new mat[layers];
	mat *Bias = new mat[layers];
	std::string *ActFunc = new std::string[layers];
	int *hiddenDim = new int[layers+1];


	hiddenDim[0] = inputDim;
	hiddenDim[1] = MAX_HID_NODE/numprocs;

	 if(MAX_HID_NODE/numprocs*numprocs != MAX_HID_NODE && myid == numprocs-1){
		hiddenDim[1] = (MAX_HID_NODE-MAX_HID_NODE/numprocs*numprocs)+hiddenDim[1];
	 }
		 


	for(int i = 0;i < layers; i++){
		InputWeight[i] = randu(hiddenDim[i+1],hiddenDim[i]) * 2 - 1;
		Bias[i] = randu(hiddenDim[i+1],1);

	}
	
	//broadcast to each MPI node

	//下面的过程是在本届点的算法，每个节点都会执行该程序
	mat myK,H;
	mat beta;
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&elm_start,0);

	run_ELM(TrainData,TrainLabels,H,myK,InputWeight,Bias, ActFunc,C,layers,NumberClasses);

	gettimeofday(&elm_stop,0);

	timeuse = 1000000 * (elm_stop.tv_sec - elm_start.tv_sec) + (elm_stop.tv_usec - elm_start.tv_usec);
	elm_time = timeuse / 1000000;

	//double err_rate = ELM_test(TestData, TestLabels, InputWeight, beta, Bias, ActFunc, C, layers, NumberClasses);
	//printf("myid = %d, err_rate = %f\n", myid, err_rate);
	
	printf("myid = %d, Now begin reduce K!\n",myid);
	gettimeofday(&comm_time_start,0);
	double *myK_val = new double[myK.n_rows*myK.n_cols];
	double *K_val = new double[myK.n_rows*myK.n_cols];	


	for(int m = 0; m < myK.n_rows; m++){
		for(int n = 0; n < myK.n_cols; n++){
			myK_val[m*myK.n_cols+n] = myK(m,n);
		}
	}

	MPI_Reduce(myK_val,K_val,myK.n_rows*myK.n_cols,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	mat K = zeros(myK.n_rows, myK.n_cols);

			
	for(int m = 0; m < myK.n_rows; m++){
		for(int n = 0; n < myK.n_cols; n++){
			K(m,n) = K_val[m*myK.n_cols+n] ;
		}
	}
	printf("myid = %d, Now end reduce K!\n",myid);
	gettimeofday(&comm_time_stop,0);
	timeuse = 1000000 * (comm_time_stop.tv_sec - comm_time_start.tv_sec) + (comm_time_stop.tv_usec - comm_time_start.tv_usec);
	comm_time += timeuse / 1000000;

	

	mat S = zeros(K.n_rows,K.n_cols);
	double *S_val = new double[K.n_rows*K.n_cols];
	gettimeofday(&elm_start,0);
	if(myid == 0){
		S = inv(K+eye(S.n_rows,S.n_cols)/C);
	}

	gettimeofday(&elm_stop,0);
	timeuse = 1000000 * (elm_stop.tv_sec - elm_start.tv_sec) + (elm_stop.tv_usec - elm_start.tv_usec);
	combine_time += timeuse / 1000000;

	gettimeofday(&comm_time_start,0);
	if(myid == 0){
		
		for(int m = 0; m < S.n_rows; m++){
			for(int n = 0; n < S.n_cols; n++){
				S_val[m*S.n_cols+n] = S(m,n);
			}
		}
		MPI_Bcast(S_val,S.n_rows*S.n_cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
	}else{
		MPI_Bcast(S_val,S.n_rows*S.n_cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
		for(int m = 0; m < S.n_rows; m++){
			for(int n = 0; n < S.n_cols; n++){
				S(m,n) = S_val[m*S.n_cols+n] ;
			}
		}
	}
	printf("myid = %d, Now end MPI_Bcast K!\n",myid);
	gettimeofday(&comm_time_stop,0);
	timeuse = 1000000 * (comm_time_stop.tv_sec - comm_time_start.tv_sec) + (comm_time_stop.tv_usec - comm_time_start.tv_usec);
	bcast_time += timeuse / 1000000;


	gettimeofday(&elm_start,0);

	mat Target = onehot(NumberClasses, NumberSamples, TrainLabels);

	beta = H*S*Target.t();
	
	gettimeofday(&elm_stop,0);
	timeuse = 1000000 * (elm_stop.tv_sec - elm_start.tv_sec) + (elm_stop.tv_usec - elm_start.tv_usec);
	elm_time += timeuse / 1000000;	
	printf("myid = %d, Now end beta!\n",myid);	
	

	gettimeofday(&all_time_stop,0);
	timeuse = 1000000 * (all_time_stop.tv_sec - all_time_start.tv_sec) + (all_time_stop.tv_usec - all_time_start.tv_usec);
	all_time = timeuse / 1000000;



	//begin testing


	gettimeofday(&comm_time_start,0);
	printf("myid = %d, Now start testing!\n",myid);	
	int NumberTestData = TestData.n_cols;
	mat testH = TestData;
	int NumberHiddenUnits = TestData.n_rows;
	for(int layer_id = 0; layer_id < layers; layer_id++){
		NumberHiddenUnits = InputWeight[layer_id].n_rows;
		testH = ActivateFunction(ActFunc[layer_id], InputWeight[layer_id] * testH + repmat(Bias[layer_id],1,NumberTestData));
	}
	mat output = (testH.t() * beta).t();

	double *output_val = new double[output.n_rows*output.n_cols];
	double *alloutput_val = new double[output.n_rows*output.n_cols];
	for(int m = 0; m < output.n_rows; m++){
		for(int n = 0; n < output.n_cols; n++){
			output_val[m*output.n_cols+n] = output(m,n);
		}
	}

	MPI_Reduce(output_val,alloutput_val,output.n_rows*output.n_cols,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(myid == 0){
		for(int m = 0; m < output.n_rows; m++){
			for(int n = 0; n < output.n_cols; n++){
				output(m,n) = output_val[m*output.n_cols+n] ;
			}
		}
		mat max_vals = max(output);

		mat pred_labels = zeros(NumberTestData,LABELDIM);

		int sum = 0;
		for(int i = 0;i < NumberTestData; i++){
			pred_labels[i] = 0;
			for(int j = 0;j < NumberClasses; j++){
				if(max_vals(i) == output(j,i)){
					pred_labels[i] = j+1;
					continue;
				}
			}
			if(pred_labels(i) != TestLabels(i)){
				sum ++;
			}
		}
		double predacc = (1-(double)sum/NumberTestData)*100;
		printf("testing accurate = %f%%\n", predacc);
	}
	


	delete []S_val;
	delete []K_val;
	//double err_rate = ELM_test(TestData, TestLabels, InputWeight, beta, Bias, ActFunc, C, layers, NumberClasses);

	//printf("err_rate = %f\n", err_rate);
	




	printf("myid = %d, The bcast proc costs %f s \n", myid, bcast_time);
	printf("myid = %d, ELM costs %f s\n", myid, elm_time);
	printf("myid = %d, Communicate costs %f s\n", myid, comm_time);
	printf("myid = %d, Combine time costs %f s\n", myid, combine_time);
	printf("myid = %d, The trst proc costs %f s \n", myid, test_time);
	printf("myid = %d, The whole program costs %f s \n", myid, all_time);
	

	delete []InputWeight;
	delete []Bias;
	delete []ActFunc;
	delete []hiddenDim;


	MPI_Finalize();  



	return 0;
}
