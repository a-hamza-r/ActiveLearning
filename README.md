# ActiveLearning
An Active Learning Algorithm on two datasets MMI and MindReading

You should be able to find a 'main.py' file in 'Code Files Python' folder. 
I used python3 for writing the code. 

I used some necessary libraries, which include: 
1) loadmat from scipy.io 
2) os 
3) LogisticRegression from sklearn.linear_model
4) statistics
5) math
6) random
7) numpy
8) matplotlib.pyplot
please have them installed before running the code.

I have added all the test cases to main.py. If you run main.py using python3, it should automatically run them. It first loads data for MindReading dataset, applies active learning algorithm on it (using both random and uncertainty based sampling) and then draws a graph for it. Then it does the same for MMI dataset. The comments should be able to guide you how the test cases (and rest of the code) work. 

Note: Please do not change the names of directories and files used during the assignment as the code depends on the specific naming convention used. However, you can change the data in the files and it should still work perfectly fine.