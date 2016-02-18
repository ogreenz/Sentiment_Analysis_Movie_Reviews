function [class_vec] = filelist_to_class_vec(filelist)
%file list contains the full path if we train the convention is to put
%[pos/neg]_id_rating.txt where pos corresponds to +1 and neg to -1. this
%function returns the class vectors to mathcing the filelist.
[n, ~] = size(filelist);
class_vec = zeros(1,n); %column vec
for i=1:n
    [~,filename,~] = fileparts(filelist(i).name);
    if strncmpi(filename,'pos',3)
        class_vec(i) = 1;
    else
        class_vec(i) = -1;
    end
end
end