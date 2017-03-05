function test_results = autocnn_prediction(data_test, net, opts, model)
fprintf('setting up GPU and %s \n', upper('dependencies'))
[net, opts] = set_up(net, opts); % check GPU, dependencies and add paths
net.layers{1}.flip = false;
opts.norm = 'stat';

if ~isfield(data_test,'labels') || isempty(data_test.labels)
    data_test.labels = zeros(size(data_test.images,1),1);
end

if (net.layers{1}.augment)
  images = flip(reshape(data_test.images,[size(data_test.images,1),net.layers{1}.sample_size]),3);
  data_test.images = cat(1,data_test.images,reshape(images,[],prod(net.layers{1}.sample_size)));
end

fprintf('\n-> processing %s samples \n', upper('test'))
if (net.layers{1}.crop)
  % take 4 corner crops + 1 central
  test_features = {};
  offsets = [1,net.layers{1}.sample_size(1)-net.layers{1}.crop];
  for row = offsets
    for col = offsets
      net.layers{1}.crop_offset = [row,col];
      test_features{end+1} = forward_pass(data_test.images, net);
    end
  end
  net.layers{1}.crop_offset = round([row/2,col/2]); % central crop
  test_features{end+1} = forward_pass(data_test.images, net);
  test_features = cat(1,test_features{:});
  net.layers{1}.crop_offset = 0;
else
  test_features = forward_pass(data_test.images, net);
end

%% Classification
fprintf('\n-> %s with %s \n', upper('Prediction'), upper(opts.classifier));
scores = {};
for j=1:numel(model)
    test_data_dim = test_features(:,1:opts.PCA_dim(j));
    if (~isempty(opts.norm))
        test_data_dim = feature_scaling(test_data_dim, opts.norm);
    end
    scores{end+1} = predict_batches(test_data_dim, repmat(data_test.labels,size(test_data_dim,1)/length(data_test.labels),1), data_test.labels, unique(data_test.labels), model{j}, @predict, opts);
end
scores_all = mean(cat(3,scores{:}),3);
test_results.scores = zeros(size(scores_all));
for i=1:length(model{1}.Label)
    test_results.scores(:,model{1}.Label(i)+1) = scores_all(:,i);
end
[~,idx] = max(test_results.scores,[],2);
idx = idx-1;
test_results.predicted_labels = {idx}; % in cell to keep consistency with other code
test_results.scores = {test_results.scores};
test_results.acc(1,1) = nnz(idx == data_test.labels)/numel(data_test.labels)*100;
test_results.acc(2,1) = test_results.acc(1,1);
fprintf('Accuracy of a single classifier model = %f (%d/%d)\n', test_results.acc(1), nnz(idx == data_test.labels), length(data_test.labels))

end

function [net, opts] = set_up(net, opts)
try
    D = gpuDevice(1);
    g = gpuArray(rand(100,100));
    fprintf('GPU is OK \n')
    if (~isfield(opts,'gpu')), for k=1:numel(net.layers), net.layers{k}.gpu = true; end; end
catch e
    warning('GPU not available: %s', e.message)
    for k=1:numel(net.layers), net.layers{k}.gpu = false; end
end
if (isfield(opts,'matconvnet') && exist(opts.matconvnet,'dir'))
    addpath(fullfile(opts.matconvnet,'matlab/mex'))
    run(fullfile(opts.matconvnet,'matlab/vl_setupnn.m'))
    vl_nnconv(rand(32,32,3,10,'single'),rand(5,5,3,20,'single'),[]);
    try
        vl_nnconv(gpuArray(rand(32,32,3,10,'single')),gpuArray(rand(5,5,3,20,'single')),[]);
    catch e
        warning('MatConvNet: GPU not available: %s', e.message)
        for k=1:numel(net.layers), net.layers{k}.gpu = false; end
    end
    fprintf('MatConvNet is OK \n')
else
    warning('MatConvNet not found, Matlab implementation will be used')
    for k=1:numel(net.layers), net.layers{k}.is_vl = false; end
end
if (isfield(opts,'gtsvm') && exist(opts.gtsvm,'dir'))
    addpath(opts.gtsvm)
    % check that it works
    context = gtsvm;
    context.initialize( rand(1000,100), randi([0 4],1000,1), true, 1, 'gaussian', 0.05, 0, 0, false );
    context.optimize( 0.01, 1000000 );
    classifications = context.classify( rand(1000,100) );
    opts.classifier = 'gtsvm';
    fprintf('GTSVM is OK \n')
elseif (isfield(opts,'libsvm') && exist(opts.libsvm,'dir'))
    addpath(opts.libsvm)
    svmtrain(randi(5,100,1),rand(100,100),'-q'); % check that it works
    opts.classifier = 'libsvm';
    fprintf('LIBSVM is OK \n')
elseif (isfield(opts,'liblinear') && exist(opts.liblinear,'dir'))
    addpath(opts.liblinear)
    train(randi(5,100,1),sparse(rand(100,100)),'-q'); % check that it works
    opts.classifier = 'liblinear';
    fprintf('LIBLINEAR is OK \n')
else
    opts.classifier = 'lda';
    warning('LIBSVM, LIBLINEAR or GTSVM should be installed, Matlab LDA implementation will be used for classification')
end

if (~isfield(opts,'test_path'))
  if (opts.val)
    opts.test_path = fullfile(opts.dataDir,'val_results');
  else
    opts.test_path = fullfile(opts.dataDir,'test_results');
  end
end
if (~exist(opts.test_path,'dir'))
    mkdir(opts.test_path)
end
addpath(opts.test_path)

end