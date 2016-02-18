function [words_map] = filelist_to_words_map(words_map, file_list)
% iterates through the file list reviews and adds the the existing word_map
% and word_count unique words.
% IMPORTANT: file list contains the full path to the file.
files = file_list;

[n,~] = size(files);
words_count = 1;
str = sprintf('Number of documents: %d\n', n);
disp(str);
for i = 1 : n
   if (mod(i,100) == 0)
       str = sprintf('i is now %d\n', i);
       disp(str);
   end
   doc_path = files(i).name;
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
         if (~isKey(words_map, word))
            words_map(char(word)) = words_count;
            words_count = words_count + 1;
         end
      end
      line = fgets(fid);
   end
   fclose(fid);
end

end