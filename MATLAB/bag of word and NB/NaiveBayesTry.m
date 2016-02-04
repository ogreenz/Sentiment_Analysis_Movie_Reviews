[data_mat, words_map] = bag_of_words(10, 0.4);
[m, n] = size(data_mat);

% Shuffle the data
ordering = randperm(m);
data_mat = data_mat(ordering,:);

% Cross validate
par = 10;
accuracy_sum = 0;
for k = 1 : par
   low = round((k-1)*(m/par)) + 1;
   high = round(k*(m/par));
   train_dat = [data_mat(1:(low-1), 1:(n-1)); data_mat(high+1:m, 1:(n-1))];
   train_lab = [data_mat(1:(low-1), n); data_mat(high+1:m, n)];
   test_dat = data_mat(low:high, 1:(n-1));
   test_lab = data_mat(low:high, n);
   
   % Fit the data to the naive bayes model (make sure all the vars aren't 0
   pos_bool = zeros(1, n-1);
   neg_bool = zeros(1, n-1);
   last_pos_idx = 0;
   last_neg_idx = 0;
   [tn, ~] = size(train_dat);
   for i = 1 : tn
       for j = 1 : n-1
           if (train_lab(i) == 1)
               if (train_dat(i, j) ~= 0)
                   pos_bool(j) = 1;
               end
               last_pos_idx = i;
           else
               if (train_dat(i, j) ~= 0)
                   neg_bool(j) = 1;
               end
               last_neg_idx = i;
           end  
       end
   end
   for i = 1 : n-1
       if (pos_bool(i) == 0)
           train_dat(last_pos_idx, i) = 1;
       end
       if (neg_bool(i) == 0)
           train_dat(last_neg_idx, i) = 1;
       end
   end

   Mdl = fitcnb(train_dat,train_lab);
   res_labels = predict(Mdl, test_dat);
   [count, ~] = size(test_lab);
   error = 0;
   for i = 1 : count
      if (test_lab(i) ~= res_labels(i))
          error = error + 1;
      end
   end
   cur_acc = 1 - error/count;
   accuracy_sum = accuracy_sum + cur_acc;
   str = sprintf('The accuracy of index %d: %f', k, cur_acc);
   disp(str);
end

accuracy = accuracy_sum/par;
str = sprintf('The total accuracy: %f', accuracy);
disp(str);