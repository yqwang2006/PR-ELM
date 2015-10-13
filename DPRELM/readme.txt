首先根据不同数据集调整代码中部分参数
例如POSELM.h中的
#define DATADIM attributes+1 
attribute指数据维度

再在main.cpp调整C值和hiddenNodeNum值、ClassNumber值

之后编译
./make.sh

编译后得到bin/main，将该文件拷贝到各个节点上
source ./scpmain
输入各个节点密码

再将分割好的所有数据集拷贝到s0节点，之后运行
./transferdata.sh
输入各个机器的密码

最后运行
4节点：
mpirun -l -n 1 -host s0 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s1 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s2 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host ubuntu17-05 /home/wyq/MPIELM_N_to_1_gather/bin/main
2节点：
mpirun -l -n 1 -host s0 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s1 /home/wyq/MPIELM_N_to_1_gather/bin/main
