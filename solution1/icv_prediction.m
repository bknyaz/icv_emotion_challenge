function test_results = icv_prediction(test_dir, test_img_list, model_file, submission_file)

test_results = load(model_file)

im_size = [96,96];
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
data_submit.labels = zeros(size(data_submit.images,1),1);
data_submit.users = unique(users)';
data_submit.images = single(data_submit.images)./255  % print submission data

scores = {};
for m=1:numel(test_results.scores) % collect SVM scores from all models
    results = autocnn_prediction(data_submit, test_results.net{m}, test_results.opts, test_results.model{m})
    scores{m} = results.scores{1};
end
icv_write_submission(mean(cat(3,scores{:}),3), submission_file)

end