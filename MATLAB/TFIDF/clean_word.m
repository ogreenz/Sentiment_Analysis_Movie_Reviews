function [cleaned_word] = clean_word(word)

% Check that this is not some html garbage
[m, ~] = size(strfind(char(word),'><'));
if (m > 0)
    cleaned_word = '';
    return;
end
          
word = lower(word);
word = regexprep(word, '[?.`,-!-()<>/\\]', '');
word = char(word);
if (isempty(word) || max(~isstrprop(word,'alpha')))
	cleaned_word = '';
    return;
end
cleaned_word = word;

end