# Predict-the-user-intention-with-Support-Vector-Machines-and-SVMBM25
Background
The intent analysis is one of the important fields in Natural Language Processing. By analyzing the pattern of the text, the machine can understand what the user needs most at that time. The intent analysis can be used when building the chatbot, auto driving car, banking, and financial system, etc.
Although all sentence contains the intention, not every sentence is important. For example in the mailing system, some emails can be detected as the Spam because the content of the email are have no clear intention. Another problem in intent analysis is the bias. Even for a human, analyzing the intention of others is tricky and sometimes can result in the multiple interpretations. These two problems motivated the NLP researcher to build the intent analysis method to minimize the error.
We suggest the intent classification using the BM25 and SVM.  Okapi BM25 is a ranking function which represents almost similar to the TF-IDF during the document retrieval. We found that this method has high accuracy (more than 95%) in the classification. 

Method:
1. Get the BM25 score of the test and train files
2. Scaling the score into [0..1] range to save the computation time
3. Train the SVM to classify the intention
4. Predict the intention of the test files, and count the accuracy score.

Tools:
1. Environment: Ubuntu 16.04 LTS, memory 4GB
2. Platform: Code Block 16.01
3. Dataset: NLU Evaluation Corpora dataset (using only 7 categories)
4. Classifying and evaluating the accuracy score (LibSVM and LibLinear)

Reference:
1. Dataset: https://github.com/sebischair/NLU-Evaluation-Corpora
2. Okapi BM25 concept: Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze. An Introduction to Information Retrieval, Cambridge University Press, 2009, p. 233
3. LibSVM: hih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011
4. LibLinear: R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A Library for Large Linear Classification, Journal of Machine Learning Research 9(2008), 1871-1874. Software available at http://www.csie.ntu.edu.tw/~cjlin/liblinear

How to use and test:
1. Put the intent category in train_file and test_file
2. Open the terminal and set the directory into the current path
3. Run the program ./SVMBM25
4. All the results are stored in the output folder (BM25 score, BM25 after scaling, SVM model, test file prediction)

Notes:
This method can also detect the sentence which has the weak intention (in this program we put it in none.txt). To make this more accurate, we suggest more samples for the weak intention in training sentence than other categories. 
