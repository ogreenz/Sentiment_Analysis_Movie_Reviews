function [vec] = doc_to_vec(words_map, doc_path)

[m, ~] = size(words_map);
vec = zeros(1, m);
fid = fopen(doc_path);
line = fgets(fid);
while (ischar(line))
   words = strsplit(line);
   [~, words_num] = size(words);
   for j = 1 : words_num - 1
       word = words(j);
       word = clean_word(word);
       if (isempty(word))
            continue;
       end
       if (isKey(words_map, word))
           ind = words_map(word);
           vec(ind) = vec(ind) + 1;
       end
   end
   line = fgets(fid);
end
fclose(fid);

end