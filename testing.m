% Target Character Prediction (New User) part of the paper titled:
% "Transfer Learning of an Ensemble of DNNs for SSVEP BCI Spellers without
% User-Specific Training"
%% Preliminaries
% Please download benchmark [1] and/or BETA [2] datasets
% and add folder that contains downloaded files to the MATLAB path.

% [1] Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
% ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and
% Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.

% [2] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
% benchmark database toward ssvep-bci application,” Frontiers in
% Neuroscience, vol. 14, p. 627, 2020.
%% Specifications (e.g. number of character) of datasets
subban_no=3; % # of subbands/bandpass filters
dataset='Bench'; % 'Bench' or 'BETA' dataset
signal_length=0.4; % Signal length in second
if strcmp(dataset,'Bench')
    totalparticipants=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length
    total_ch=64; % # of channels used at collection of the dataset
    max_epochs=500; % # of epochs for first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
    addpath('C:\Users\bg060\Documents\MATLAB\SSVEP\Benchmark Dataset');
    load('Freq_Phase.mat')
elseif strcmp(dataset,'BETA')
    totalparticipants=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    sample_length=sampling_rate*signal_length; %
    total_ch=64;
    max_epochs=800;
    dropout_second_stage=0.7;
    %C:\Users\bg060\Documents\MATLAB\SSVEP\BETA Dataset
    addpath('C:\Users\bg060\Documents\MATLAB\SSVEP\BETA Dataset');
    load('Freqs_Beta.mat')
    %else %if you want to use another dataset please specify parameters of the dataset
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end
%% Preprocessing
total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
[AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalparticipants,totalblock,totalcharacter,sampling_rate,dataset);
% Dimension of AllData:y_AllData
% (# of channels, # sample length, #subbands, # of characters, # of blocks, # of subjects)
%% Construct the artificial reference signals using 5 harmonics as defined in Equation 1 of [4].
% [4] Z. Lin, C. Zhang, W. Wu, and X. Gao, “Frequency recognition based on
% canonical correlation analysis for ssvep-based bcis,” IEEE Transactions
% on Biomedical Engineering, vol. 53, no. 12, pp. 2610–2614, 2006

ref_signals=zeros(10,sample_length,totalcharacter);
t= (0):(1/250):(signal_length-1/250);
for i=1:totalcharacter
    tmp_ref = [ sin(2*pi*t*freqs(i));...
        cos(2*pi*t*freqs(i));...
        sin(4*pi*t*freqs(i));...
        cos(4*pi*t*freqs(i));...
        sin(6*pi*t*freqs(i));...
        cos(6*pi*t*freqs(i));...
        sin(8*pi*t*freqs(i));...
        cos(8*pi*t*freqs(i));...
        sin(10*pi*t*freqs(i));...
        cos(10*pi*t*freqs(i))];
    ref_signals(:,:,i)=tmp_ref;
end
%% Target recognition of our method for an SSVEP EEG speller instance x
sizes=size(AllData);
total_ins=totalcharacter*totalblock; % Total number of instances that the test participant has

acc_matrix=zeros(totalparticipants,totalblock);
all_final_predictions=zeros(total_ins,totalparticipants);


% Add the folder, where Ensemble of DNNs (fine-tuned participants' DNNs)
% are saved, to the MATLAB path:
% addpath('...');

% Measure the performance of our Ensemble-DNN over all the participants
% in a leave-one-participant-out fashion
for test_participant=1:totalparticipants
    AllParticipants=1:totalparticipants;
    AllParticipants(test_participant)=[]; % Get indexes of training participants
    
    
    sv_name=['participants_DNNs_',dataset,'_',num2str(signal_length),'_',int2str(test_participant),'.mat']; 
    load(sv_name);
    
    % Take the sub-band and channel combinations' weights of fine-tuned participants' DNNs:
    all_channel_combs=zeros(length(channels),120,totalparticipants-1);
    all_subband_combs=zeros(1,1,3,totalparticipants-1);
    for n = 1:totalparticipants-1
        prt = AllParticipants(n);
        subband_weights=participants_DNNs{prt, 1}.Layers(2, 1).Weights;
        all_subband_combs(:,:,:,n)=subband_weights;
        all_channel_combs(:,:,n)=squeeze(participants_DNNs{prt, 1}.Layers(3, 1).Weights);
    end
    %
    
    % Take all the instances of test participant and their labels:
    testdata=AllData(:,:,:,:,:,test_participant);
    testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter*totalblock]);
    
    test_y=y_AllData(:,:,:,test_participant);
    test_y=reshape(test_y,[1,totalcharacter*totalblock]);
    %
    
    % new user = test participant 
    
    for idx =1:total_ins
        counter =0;
        
        similarities_idx=zeros(totalparticipants-1,1); % Initialize the vector that stores the similarity measures between the new user's  instance and training participants
        predictions_idx=zeros(totalparticipants-1,1);  % Initialize the vector that stores the predictions of training participants' DNNs for the new user's instance
        
        %% Get the predictions of all the training participants for the idx'th new user instance and calculate the similarities between training participants and the new user's instance
        for n = 1:totalparticipants-1
            counter = counter+1;
            
            prt = AllParticipants(n); % Get the index of the n'th participant
            net = participants_DNNs{prt,1}; % Get the fine-tuned DNN of the n'th participant
            train_data=AllData(:,:,:,:,:,prt); % Get the all data of the n'th participant
            
            [prediction] = classify(net,testdata(:,:,:,idx)); % Get the prediction of n'th participant's DNN for the test instance
            prediction=double(prediction); % Convert prediction to double from categorical type            
            predictions_idx(counter)=prediction;
            
            test_ins=sum(all_subband_combs(:,:,:,n).*testdata(:,:,:,idx),3); % Combine the sub-bands of the new user instance using the sub-band combination weight of n'th participant's DNN           
            
            % Get the template of the n'th training participant:            
            template = mean(train_data(:,:,:,prediction,:),5);
            %
            
            template=sum(all_subband_combs(:,:,:,n).*template,3); % % Combine the sub-bands of the n'th participant's template using the sub-band combination weight of n'th participant's DNN
            
            % Calculate the similarity measure for all the channel
            % combination weights and pick the one with the maximimum
            % similarity 
            all_tmp_similarities=zeros(120,1);
            for k_ch=1:120%
                tmp_corr=corrcoef(all_channel_combs(:,k_ch,n)'*template,all_channel_combs(:,k_ch,n)'*test_ins);
                all_tmp_similarities(k_ch)=all_tmp_similarities(k_ch)+(tmp_corr(1,2)^2);
                
                [~,~,rhos]=canoncorr((all_channel_combs(:,k_ch,n)'*test_ins)',ref_signals(:,:,prediction)');
                all_tmp_similarities(k_ch)=all_tmp_similarities(k_ch)+(rhos(1)^2); %sign(rhos(1))*
                
            end
            rho_n=max(all_tmp_similarities);% The similarity between n'th participant and the instance of new user (test participant)
            similarities_idx(counter)=rho_n;
        end
        
        %% Dynamic Selection
        
        
        [sorted_similarities_idx, indeks_sorted_similarities_idx]=sort(similarities_idx,'descend');        
        
        tmp_vote_all=zeros(40,totalparticipants-1);
        
        
        for k =1:(totalparticipants-1)
            ind_array_distance = indeks_sorted_similarities_idx(1:k);
            pred_array_distance = predictions_idx(ind_array_distance);
            
            
            for k2=1:k
                tmp_vote_all(pred_array_distance(k2),k)=tmp_vote_all(pred_array_distance(k2),k)+ (sorted_similarities_idx(k2));     %(71-jjj_k);
            end
            
        end
        
        tmp_confidence=zeros((totalparticipants-1),1);
        for k=1:(totalparticipants-1)
            tmp_votes=tmp_vote_all(:,k);
            tmp_votes=sort(tmp_votes,'descend');
            tmp_confidence(k)=tmp_votes(1)-tmp_votes(2);            
        end        
        
        [~,most_confident_k]=max(tmp_confidence);
        final_prediction=tmp_vote_all(:,most_confident_k);
        [~,final_prediction]=max(final_prediction);
        all_final_predictions(idx,test_participant)=final_prediction;
           
    end
    % Calculate the accuracy on the test participant (block-wise)
    for blck=1:totalblock
        acc_matrix(test_participant,blck)= mean(all_final_predictions((blck-1)*40+1:(blck)*40,test_participant)==double(test_y(1:40)'));
    end    
end