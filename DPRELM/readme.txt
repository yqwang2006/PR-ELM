���ȸ��ݲ�ͬ���ݼ����������в��ֲ���
����POSELM.h�е�
#define DATADIM attributes+1 
attributeָ����ά��

����main.cpp����Cֵ��hiddenNodeNumֵ��ClassNumberֵ

֮�����
./make.sh

�����õ�bin/main�������ļ������������ڵ���
source ./scpmain
��������ڵ�����

�ٽ��ָ�õ��������ݼ�������s0�ڵ㣬֮������
./transferdata.sh
�����������������

�������
4�ڵ㣺
mpirun -l -n 1 -host s0 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s1 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s2 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host ubuntu17-05 /home/wyq/MPIELM_N_to_1_gather/bin/main
2�ڵ㣺
mpirun -l -n 1 -host s0 /home/wyq/MPIELM_N_to_1_gather/bin/main : -n 1 -host s1 /home/wyq/MPIELM_N_to_1_gather/bin/main
