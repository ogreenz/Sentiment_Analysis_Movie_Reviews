%creates the reviews file list
reviews_dir = 'all';
filelist = dir(strcat(reviews_dir,'\*.txt'));
for i=1:size(filelist,1)
    filelist(i).name = fullfile(pwd, reviews_dir, filelist(i).name);
end
%split the filelist into train reviews and test reviews using random
%permutation.
train_file_list = ..;
test_file_list = ..;

%create train tf-idf matrix
words_map = containers.Map();
[words_map, words_count] = filelist_to_words_map(words_map, train_file_list);
train_count_matrix = filelist_to_mat(train_file_list, words_map);
[train_count_matrix, words_map] = limit_features(train_count_matrix, words_map, 1500);
idf_vector = idf_vectorizer(train_count_matrix);
train_tf_idf_matrix = idf_transform(idf_vector, train_count_matrix);
train_tf_idf_matrix_labels = filelist_to_class_vec(train_file_list);
%create test tf-idf matrix
%note the the words map is already created using train and limited
test_count_matrix = filelist_to_mat(test_file_list, words_map);
test_tf_idf_matrix = idf_transform(idf_vector, test_count_matrix);
%train and test model here
