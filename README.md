# Transfer Learning of an Ensemble of DNNs for SSVEP BCI Spellers without User-Specific Training
This is the official repository for the paper [1] titled "An Ensemble of DNNs for SSVEP BCI Spellers without User-Specific Training". This repository allows you to train the ensemble of DNNs and classify the SSVEP signal using the most similar $k$ many DNNs as explained in the paper.

# Preparation
The Benchmark dataset [1] and BETA dataset [2] must be downloaded. The link for the both datasets: http://bci.med.tsinghua.edu.cn/download.html.

# Training and evaluating the proposed Ensemble of DNNs method
In our performance evaluations, we conducted the comparisons (following the procedure in the literature) in a leave-one-participant-out fashion.
For example, we constitute the ensemble using 34 (69) participants and test the performance on the remaining participant that is considered as new user. This process is repeated 35 (70) times in the case of the benchmark (BETA) dataset. While calculating the information transfer rate (ITR) results, a 0.5 seconds gaze shift time is taken into account. 

on 5 (or 3) and test on the remaining block and repeat this process 6 (4) times in order to have exhaustively tested on each block in the case of the benchmark (or the BETA) dataset. For fairness, we take into account a 0.5 seconds gaze shift time while computing the ITR results (as it is computed in other methods). We test with the pre-determined set of 9 channels (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2) again for fair comparisons (since these are the channels that have been used in the compared methods), but we also test with all of the available 64 channels to fully demonstrate the efficacy of our DNN. 

# References 
1. Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
   ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
   Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.
2. B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
   benchmark database toward ssvep-bci application,” Frontiers in
   Neuroscience, vol. 14, p. 627, 2020.
