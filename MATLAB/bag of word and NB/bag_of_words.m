function [data_mat, words_map] = bag_of_words(min_app, alpha)
% neg_dir = 'check_neg';
% pos_dir = 'check_pos';
neg_dir = 'neg';
pos_dir = 'pos';
words_map = containers.Map();
words_count = 1;
[words_map, words_count] = create_words_map(words_map,words_count,neg_dir);
[words_map, ~] = create_words_map(words_map,words_count,pos_dir);
words_map = reduce_dim(words_map, pos_dir, neg_dir, min_app, alpha);
neg_data_mat = dir_to_mat(neg_dir, words_map, -1);
pos_data_mat = dir_to_mat(pos_dir, words_map, 1);
data_mat = [neg_data_mat; pos_data_mat];
end