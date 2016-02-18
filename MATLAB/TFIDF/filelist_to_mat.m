function [data_mat] = filelist_to_mat(filelist, words_map)
%the files in the list should be reviews and full path.  this will create the sparse
%matrix.
cols = size(words_map,1);
files = filelist;
[n,~] = size(files);
rows = n;

str = sprintf('creating a matrix of size %d X %d', rows, cols);
disp(str);
%data_mat = zeros(rows, cols);
data_mat = sparse(rows, cols);
doc_ind = 1;
for i = 1 : n
   if (mod(i,100) == 0)
       str = sprintf('in dir to mat of %s, i is now %d\n', file_dir, i);
       disp(str);
   end
   doc_path = files(i).name;
   vec = doc_to_vec(words_map, doc_path);
   data_mat(doc_ind, 1:cols) = vec;
   doc_ind = doc_ind + 1;
end
end