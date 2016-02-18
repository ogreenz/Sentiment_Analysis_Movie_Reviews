function [idf_vector] = idf_vectorizer(count_matrix)
%TFIDF transformer fit function.

    [n_samples, n_features] = size(count_matrix);   
    bin_count = histc(count_matrix, 1:n_features,1);
    document_frequency = bin_count(1,:);
    
    %smooth idf
    document_frequency = document_frequency + 1;
    n_samples = n_samples + 1;
    idf_vector = log(n_samples./document_frequency) + 1;
end
    