依赖的库和环境：pytorch、pandas、numpy、tqdm、scipy

代码结构：  
1、Train_Val3.py：模型训练文件，但是超参数已经有所变动，不能保证训练出和leaderboard上的最好成绩一样的效果。  
2、test.py：测试文件，并输出预测结果"submission.csv"。  
3、Model.py：模型设置文件。  
4、Dataset_process.py：包含数据读入和预处理函数的文件。  
5、model文件夹：包含了得到最好测试结果的模型（共三个模型文件）以及模型在测试集上的结果submission.csv。  
6、data文件：从kaggle上下载得到的全部数据。  

test.py代码运行：  
1、需要将M3DV下的所有文件都下载到一个目录下，以正常运行test.py文件。  
2、test.py需要读入三个文件，分别是：(1)data/test文件夹下的数据；(2)data文件夹中的sampleSubmission.csv文件；(3)model文件夹下的三个模型。当然文件路径可以修改。  
3、确认文件路径后直接运行test.py。  
4、test.py运行后会在代码文件所在的目录下生成submission.csv文件，文件结果应与model文件夹下的submission.csv文件内容一致。
