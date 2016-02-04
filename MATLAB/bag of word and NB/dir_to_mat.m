function [data_mat] = dir_to_mat(file_dir, words_map, class)

cols = size(words_map,1) + 1;
files = dir(file_dir);
[n,~] = size(files);
rows = n - 2; % Ignore the '.' and '..'

str = sprintf('creating a matrix of size %d X %d', rows, cols);
disp(str);
data_mat = zeros(rows, cols);
doc_ind = 1;
for i = 1 : n
   doc_name = files(i).name;
   doc_path = strcat(file_dir,'/',doc_name);
    if (strcmp(doc_name,'.') == 1 || strcmp(doc_name,'..') == 1)
       continue;
   end
   vec = doc_to_vec(words_map, doc_path);
   data_mat(doc_ind, 1:(cols-1)) = vec;
   data_mat(doc_ind, cols) = class;
   doc_ind = doc_ind + 1;
end

end