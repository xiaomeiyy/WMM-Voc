 function [x_5,y_5] = num5instance(x,y,num_ins)
    dim_fea = size(x.tr,2);
    
    labels = unique(y.tr);
    num_labels = numel(labels);
    x_5.te = x.te;
    y_5.te = y.te;
    x_5.tr = zeros(num_labels * num_ins,dim_fea);
    y_5.tr = zeros(1, num_labels * num_ins);
    
    for i = 1 : num_labels
        labels_ins = find(y.tr == labels(i));
        start_idx = (i - 1) * num_ins +1;
        end_idx = i * num_ins;
        x_5.tr(start_idx:end_idx,:) = x.tr(labels_ins(1:num_ins),:);
        y_5.tr(:,start_idx:end_idx) = y.tr(:,labels_ins(1:num_ins));
    end
 
 end