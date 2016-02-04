function [words_map, words_count] = create_words_map(words_map, words_count, file_dir)

files = dir(file_dir);
[n,~] = size(files);

str = sprintf('Number of documents: %d\n', n);
disp(str);
for i = 1 : n
   if (mod(i,100) == 0)
       str = sprintf('i is now %d\n', i);
       disp(str);
   end
   doc_name = files(i).name;
   doc_path = strcat(file_dir,'/',doc_name);
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