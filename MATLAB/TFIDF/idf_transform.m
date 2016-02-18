function [tf_idf_matrix] = idf_transform(idf_vector, count_matrix)
% this function receives the idf vector learned and a count_matrix
% appropriate to the length of the idf vector and transforms it into a
% tf-idf representation.
[~, n_features] = size(count_matrix);
[~,n_features_idf_vector] = size(idf_vector);
tf_idf_matrix = NaN;
if n_features == n_features_idf_vector
    tf_idf_matrix = count_matrix * diag(idf_vector);
else
    printf('Error: number of features of the count matrix does not fit the number of features of the idf vector.');
end
end