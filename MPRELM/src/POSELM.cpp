#include "POSELM.h"
/**
Input:
TrainData: 784*5000, 行数表示数据维度，列数表示数据个数
TrainLabels: 5000*1, values from 1 to N, don't start from 0!
InputWeight: 随机生成的输入矩阵
Bias：随机生成的偏执值
C： 参数C
Output：
OutputWeight：计算得到的输出矩阵
HH：计算得到的特征矩阵HH=H'*H
**/
void run_ELM(mat TrainData,mat TrainLabels, mat &H, mat &K, mat *InputWeight, mat *Bias, std::string *ActivationFunction, double C, int layers, int NumberClasses ){
	printf("Now in POSELM function!\n");
	int NumberTrainData = TrainData.n_cols;

	H = TrainData;
	int NumberHiddenUnits = TrainData.n_rows;
	for(int layer_id = 0; layer_id < layers; layer_id++){
		NumberHiddenUnits = InputWeight[layer_id].n_rows;
		H = InputWeight[layer_id] * H + repmat(Bias[layer_id],1,NumberTrainData);
		H = ActivateFunction(ActivationFunction[layer_id],H);
	}

	K = H.t() * H;
	printf("Now exit from POSELM function!\n");
}

void combine_model(mat &combine_weight, mat &combine_HH, mat combine_HT, double C){

	int NumberHiddenUnits = combine_HH.n_rows;
	combine_weight = inv(combine_HH + eye(NumberHiddenUnits,NumberHiddenUnits)/C) * combine_HT;
}






//行表示类别数目，列数表示样本量
//返回值为标签对应的[-1,-1,1,-1,-1]
mat onehot(int rows, int cols, mat labels){
	mat desired_out = -1 * ones(rows,cols);

	for(int i = 0;i < cols; i++){
		if(labels(i) == 0) labels(i) = rows;
		desired_out(labels(i)-1,i) = 1;
	}

	return desired_out;
}



/*
激活函数
*/
mat ActivateFunction(std::string ActivationFunction, mat x){
	if (ActivationFunction == "line" || ActivationFunction == "linear")
		return linear(x);
	else if(ActivationFunction == "tan" || ActivationFunction == "tanh")
		return tanh(x);
	else if(ActivationFunction == "rect" || ActivationFunction == "rectifier")
		return rectifier(x);
	else 
		return sigmoid(x);
}


mat sigmoid(const mat x){
	mat y = zeros(x.n_rows, x.n_cols);
	for(int i = 0;i < x.size();i++){
		y(i) = 1/(1+std::exp(-x(i)));
	}
	return y;
}
/*
激活函数
*/
mat tanh(const mat x){
	mat y = zeros(x.n_rows, x.n_cols);
	for(int i = 0;i < x.size();i++){
		y(i) = std::tanh(x(i));
	}
	return y;
}
/*
激活函数
*/
mat rectifier(const mat x){
	mat y = zeros(x.n_rows, x.n_cols);
	for(int i = 0;i < x.size();i++){
		y(i) = std::max<double>(0.0,x(i));
	}
	return y;
}
/*
激活函数
*/
mat linear(const mat x){
	return x;
}
/*
load data from files
*/
void read_dat_file ( mat &Data, mat &Labels, const char *filename, int data_dim ){

	FILE *fp = fopen(filename, "rb");
	//Get the total number of data;

	fseek(fp,0,SEEK_END);
	int total_num = ftell(fp) / sizeof(double);
	rewind(fp);

	int M = total_num / data_dim;

	double *coldata = new double[M];

	Data.set_size(M,data_dim-LABELDIM);


	//labels located at the first col
	int reallen = fread(coldata, sizeof(double), M, fp);
	if(reallen < M){
		printf("read labels error!\n");
		exit(-1);
	}

	Labels.set_size(M,LABELDIM);
	for(int j = 0; j < M; j++){
		Labels(j) = coldata[j];
	}

	for(int i = 0; i < data_dim-LABELDIM; i++)
	{
		int reallen = fread(coldata, sizeof(double), M, fp);
		if(reallen < M){
			printf("reallen = %d \n",reallen);
			printf("Have done!\n");
			break;
		}
		for(int j = 0; j < M; j++){

			Data(j,i) = coldata[j];
		}
	}

	fclose(fp);
	Data = Data.t();
	delete []coldata;
}
