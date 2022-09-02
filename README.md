# Transfer Learning of an Ensemble of DNNs for SSVEP BCI Spellers without User-Specific Training
This is the official repository for the paper titled "An Ensemble of DNNs for SSVEP BCI Spellers without User-Specific Training". This repository allows you to train the ensemble of DNNs and classify the SSVEP signal using the most similar $k$ many DNNs as explained in the paper.

# Preparation
The Benchmark dataset [1] and BETA dataset [2] must be downloaded. The link for the both datasets: http://bci.med.tsinghua.edu.cn/download.html.

# Training and evaluating the proposed Ensemble of DNNs method
In our performance evaluations, we conducted the comparisons (following the procedure in the literature) in a leave-one-participant-out fashion.
For example, we constitute the ensemble using 34 (69) participants and test the performance on the remaining participant, who is considered a new user. This process is repeated 35 (70) times in the case of the benchmark (BETA) dataset. While calculating the information transfer rate (ITR) results, a 0.5 second gaze shift time is taken into account. We use the DNN architecture of [3] as a DNN architecture in the ensemble. In the DNN architecture, we use three sub-bands and nine channels (Pz, PO3, PO5,
PO4, PO6, POz, O1, Oz, O2).

# References 
1. Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
   ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
   Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.
2. B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
   benchmark database toward ssvep-bci application,” Frontiers in
   Neuroscience, vol. 14, p. 627, 2020.
3. O. B. Guney, M. Oblokulov and H. Ozkan, "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces," IEEE Transactions on Biomedical Engineering, vol. 69, no. 2, pp. 932-944,  2022.
