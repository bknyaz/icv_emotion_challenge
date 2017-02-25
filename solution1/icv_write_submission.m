function icv_write_submission(scores_all, submission_file)
[~,predicted_labels] = max(scores_all,[],2);
% write to file
fid = fopen(submission_file, 'w');
for i=1:length(predicted_labels)
    fprintf(fid, '%s\r\n', get_label_code(predicted_labels(i)-1));
end
fclose(fid);
end

function code = get_label_code(label)
if label == 0
    code = 'N_N';
else
    r = mod(label,7);
    if r == 0
        r = 7;
    end
    code = sprintf('%d_%d', (label-r)/7+1,r);
end
end