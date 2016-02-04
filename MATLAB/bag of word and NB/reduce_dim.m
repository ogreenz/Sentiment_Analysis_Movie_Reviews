function [words_map] = reduce_dim(words_map, pos_dir, neg_dir,min_app, alpha)

[m, ~] = size(words_map);
% We want to count how many time each word appears in neg/pos docs
pos_vec = zeros(1, m);
neg_vec = zeros(1, m);

pos_files = dir(pos_dir);
[pos_num,~] = size(pos_files);
neg_files = dir(neg_dir);
[neg_num,~] = size(neg_files);

str = sprintf('Number of pos documents: %d\n', pos_num);
disp(str);
str = sprintf('Number of neg documents: %d\n', neg_num);
disp(str);
for i = 1 : pos_num
   if (mod(i,100) == 0)
       str = sprintf('i is now %d\n', i);
       disp(str);
   end
   doc_name = pos_files(i).name;
   doc_path = strcat(pos_dir,'/',doc_name);
   if (strcmp(doc_name,'.') == 1 || strcmp(doc_name,'..') == 1)
       continue;
   end
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
         ind = words_map(word);
         pos_vec(ind) = pos_vec(ind) + 1;
      end
      line = fgets(fid);
   end
   fclose(fid);
end
for i = 1 : neg_num
   if (mod(i,100) == 0)
       str = sprintf('i is now %d\n', i);
       disp(str);
   end
   doc_name = neg_files(i).name;
   doc_path = strcat(neg_dir,'/',doc_name);
   if (strcmp(doc_name,'.') == 1 || strcmp(doc_name,'..') == 1)
       continue;
   end
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
         ind = words_map(word);
         neg_vec(ind) = neg_vec(ind) + 1;
      end
      line = fgets(fid);
   end
   fclose(fid);
end

total_vec = pos_vec + neg_vec;
map_keys = words_map.keys();
reverse_map = map_keys;
for i = 1 : m
    word = map_keys(i);
    ind = words_map(char(word));
    reverse_map(ind) = word;
end

% Remove unwanted words
for i = 1 : m
   total = total_vec(i);
   pos = pos_vec(i);
   if (total < min_app || abs(0.5-(pos/total)) < alpha)
      word = reverse_map(i);
      remove(words_map, word);
   end
end 

% Reset the indexes
map_keys = words_map.keys();
[~, m] = size(map_keys);
for i = 1 : m
   word = map_keys(i);
   words_map(char(word)) = i;
end

end