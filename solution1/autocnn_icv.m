function test_results = autocnn_icv(varargin)

time_start = clock;
fprintf('\ntest %s on %s \n', upper('started'), datestr(time_start))

if (nargin == 0)
    opts = [];
elseif (isstruct(varargin{1}))
    opts = varargin{1};
end

if (~isfield(opts,'whiten'))
    opts.whiten = true; % whitening is applied
end
if (~isfield(opts,'batch_size'))
    opts.batch_size = 25;
end
if (~isfield(opts,'rectifier'))
    opts.rectifier = {'relu','abs','abs'};
end
if (~isfield(opts,'conv_norm'))
    opts.conv_norm = 'rootsift';
end
if (~isfield(opts,'arch'))
    opts.arch = '1024c15-12p-conv0';
end
sample_size = [96,96,3];
opts.dataset = 'icv';
opts.net_init_fn = @() net_init(opts.arch, opts, 'sample_size', sample_size, varargin{:});
rootFolder = fileparts(mfilename('fullpath'));
if (~isfield(opts,'dataDir'))
    opts.dataDir = fullfile(rootFolder,'data/icv');
end
if (~exist(opts.dataDir,'dir'))
    mkdir(opts.dataDir)
end
fprintf('loading and preprocessing data \n')
opts.sample_size = sample_size;
if (~isfield(opts,'val') || isempty(opts.val))
  opts.val = false; % true for cross-validation on the training set
end
if (isfield(opts,'data_params') && ~isempty(opts.data_params))
    [data_train, ~, data_submit] = load_ICV_data(opts, sample_size(1:2), opts.data_params);
else
    [data_train, ~, data_submit] = load_ICV_data(opts, sample_size(1:2));
end

if (~isfield(opts,'n_folds'))
    opts.n_folds = 1;
end
if (~isfield(opts,'n_train'))
    opts.n_train = size(data_train.images,1);
end

net = opts.net_init_fn(); % initialize a network
% PCA dimensionalities (p_j) for the SVM committee
if (~isfield(opts,'PCA_dim'))
    if (numel(net.layers) > 1)
        opts.PCA_dim = [40,70,80,120,150,250,300:100:1000];
    else
        opts.PCA_dim = 50:50:500;
    end
end

for model_id = 1:opts.n_folds
    
    opts.fold_id = model_id;
    test_results = autocnn_unsup(data_train, data_submit, net, opts);
    
    fprintf('test took %5.3f seconds \n', etime(clock,time_start));
    fprintf('test (model %d/%d) %s on %s \n\n', model_id, opts.n_folds, upper('finished'), datestr(clock))
    
    time_start = clock;
end

end

function [data_train, data_test, data_submit] = load_ICV_data(opts, im_size, params)

if exist('params','var') && ~isempty(params)
    train_dir = params{1};
    training_img_list = params{2};
    test_dir = params{3};
    test_img_list = params{4};  
else
    train_dir = fullfile(opts.dataDir,'Training_dlib');
    training_img_list = fullfile(opts.dataDir,'training_new.txt');
    test_dir = fullfile(opts.dataDir,'Validation_dlib');
    test_img_list = fullfile(opts.dataDir,'order_of_validation.txt');  
end
    
if exist(fullfile(opts.dataDir,'train_96.mat'),'file') && exist(fullfile(opts.dataDir,'val_96.mat'),'file')
    data_train = load(fullfile(opts.dataDir,'train_96'));
    data_test = load(fullfile(opts.dataDir,'val_96'));
else
    fid = fopen(training_img_list);
    files_ordered = textscan(fid,'%s','Delimiter','\n');  
    fclose(fid);
    images = {};
    labels = [];
    users = [];
    for i=1:length(files_ordered{1})
        names = strsplit(files_ordered{1}{i},'\t');
        im = imread(fullfile(train_dir,names{1}));
        assert(size(im,1) == 200 && size(im,2) == 200) % images must be aligned by dlib
        if im_size(1) > 0 && im_size(1) ~= size(im,1)
            images{i} = imresize(im, im_size);
        else
            images{i} = im;
        end
        labels(i) = get_label(names{2});
        users(i) = str2double(names{1}(8:10));
    end
    
    users_u = unique(users);
    users_val = users_u(5:3:end);
    users_train = users_u(~ismember(users_u, users_val));

    images = cat(4,images{:});
    images = reshape(images, [], size(images,4))';
    train_ids = find(ismember(users,users_train));
    train_ids = train_ids(randperm(length(train_ids)));

    data_train.images = images(train_ids,:);
    data_train.labels = labels(train_ids)';
    data_train.users = users_train';  
    data_test.images = images(ismember(users,users_val),:);
    data_test.labels = labels(ismember(users,users_val))';
    data_test.users = users_val';
    assert(size(data_train.images,1)+size(data_test.images,1) == size(images,1))
%     save(fullfile(opts.dataDir,'train_96'),'-struct','data_train','-v7.3')
%     save(fullfile(opts.dataDir,'val_96'),'-struct','data_test','-v7.3')
end

data_train.images = single(data_train.images)./255;
data_test.images = single(data_test.images)./255 % print test (val) data

data_train.unlabeled_images = data_train.images;
data_train.unlabeled_images_whitened = data_train.images % print train data

if exist(fullfile(opts.dataDir,'test_96.mat'),'file')
    data_submit = load(fullfile(opts.dataDir,'test_96'));
else
    fid = fopen(test_img_list);
    files_ordered = textscan(fid,'%s','Delimiter','\n');  
    fclose(fid);
    images = {};
    users = [];
    for i=1:length(files_ordered{1})
        im = imread(fullfile(test_dir,files_ordered{1}{i}));
        assert(size(im,1) == 200 && size(im,2) == 200) % images must be aligned by dlib
        if im_size(1) > 0 && im_size(1) ~= size(im,1)
            images{i} = imresize(im, im_size);
        else
            images{i} = im;
        end
        users(i) = str2double(files_ordered{1}{i}(8:10));
    end
    images = cat(4,images{:});
    data_submit.images = reshape(images, [], size(images,4))';
    data_submit.users = unique(users)';
%     save(fullfile(opts.dataDir,'test_96'),'-struct','data_submit','-v7.3')
end
data_submit.labels = zeros(size(data_submit.images,1),1);
data_submit.images = single(data_submit.images)./255  % print submission data

% there will be no labeled validation data, so we use all training
% data for training and submission data for validation
data_train.images = cat(1,data_train.images,data_test.images);
data_train.labels = cat(1,data_train.labels,data_test.labels);
data_train.users = cat(1,data_train.users,data_test.users);
data_train.unlabeled_images = cat(1,data_train.images,data_submit.images);
data_train.unlabeled_images_whitened = cat(1,data_train.images,data_submit.images) % print train data

end

% conver to 0-49 labels
function label = get_label(code)
label = 0;
if isempty(strfind(code,'N_N'))
    label = (str2double(code(1))-1)*7+str2double(code(3));
end
end