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
opts.PCA_dim = [];
opts.arch = network_arch; %'512c15-16p-conv1_3';
opts.norm_coef = 0.25;
opts.batch_size = batch_size;
opts.test_path = '../../'; % save models in the root folder

% fixed SVM parameters found by cross-validation
opts.SVM_C = 1e-4; 
opts.SVM_B = 0;

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

% Once the models are trained and saved, they can be used as following
% load data
% data_submit = load(fullfile(opts.dataDir,'submit'));
% data_submit.images = single(data_submit.images)./255  % print submission data
% % load models
% test_results = load(test_results.test_file_name)
% % forward pass and prediction
% scores = {};
% for m=1:numel(test_results.scores)
%     results = autocnn_prediction(data_submit, test_results.net{m}, test_results.opts, test_results.model{m})
%     scores{m} = results.scores{1};
% end
% icv_write_submission(mean(cat(3,scores{:}),3), submission_file)

end