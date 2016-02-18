function [limited_count_matrix, limited_word_map] = limit_features(count_mat, word_map, limit)
% This function removes prunes rare features and leaves the limit amount of
% the most common features.
% count_mat is the matrix i- is the row which corresponds to a document j
% is the column which corresponds to a word the value in the cell in the
% number of occurences in the document.
% word map maps words -> column ind in the count matrix
% limit is the number of columns we wish to output.

% indices map is word_map inverse: column ind -> word
indices_map = containers.Map(word_map.values, words_map.keys);

limited_word_map = containers.Map();

%tfs contains the sum of all columns (frequency of the word)
tfs = sum(count_mat,1);
[~, index_map] = sort(tfs, 'descend');

% obtain only the most frequent columns
limited_count_matrix = count_mat(:,index_map);

%build the new word map (word->indice)
for i=1:limit
    word = indices_map(index_map(i));
    limited_word_map(word) = i;
end






