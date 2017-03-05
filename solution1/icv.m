function results = icv(data_dir,train_dir,training_img_list,test_dir,test_img_list, submission_file, network_arch, batch_size)
% '$DATA_DIR','$TRAINING_DIR_dlib','$TRAINING_IMG','$TEST_DIR_dlib','$TEST_IMG','$SUBMISSION_FILE'
results = [];
opts = [];
opts.vlfeat = '../../vlfeat/toolbox/mex/mexa64';
opts.matconvnet = '../../matconvnet';
liblinear_path = '../../liblinear/matlab';
run('../../vlfeat/toolbox/vl_setup')
addpath('../')

opts.dataDir = data_dir;
opts.liblinear = liblinear_path;
opts.n_folds = 5; % 5 models
opts.PCA_dim= 50:50:500; % committee of 10 PCA models
opts.arch = network_arch; %'1024c15-12p-conv0';
opts.norm_coef = 0.5;
opts.batch_size = batch_size;
opts.test_path = '../../'; % save models in the root folder

% fixed SVM parameters found by cross-validation
opts.SVM_C = 0.000005;
opts.SVM_B = 0;
opts.pca_n_samples = 30e3;

opts.data_params = {train_dir,training_img_list,test_dir,test_img_list}
%% Train a committee
test_results = autocnn_icv(opts, 'augment', true)

%% Write to the predictions.txt file
% average scores from all models
scores = {};
for m=1:numel(test_results.scores)
    scores{m} = test_results.scores{m}{1};
end
icv_write_submission(mean(cat(3,scores{:}),3), submission_file)

end