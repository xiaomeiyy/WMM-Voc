function [w0, res] = vl_WMMVoc(w0, x, class, param, awa_proto)

param.neighbour_exclude=5;
param.neighbour_nrmd = 5;
param.maximum_margin=7;
% param.epsilon=0.001;
%param.alpha=1;
%param.learningrate=0.000001;
% param.momentum=0.9;
% param.VERSION=0;
param.conf_int = 0.5;
param.d_punish = 1;
param.gain = 3;
param.lambda = 0.1;

% cal evl parameter
d = param.dis_lambda .* (log(1/(param.conf_int))).^(1./param.dis_kappa);
dX = d(class);
stand_dX = (dX/mean(dX)/param.d_punish).^param.gain;

epsilon = 0.1 * stand_dX;

kdim = size(awa_proto,2);
l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/kdim),1,size(x,2)));%球形范数�?
proto = l2norm(awa_proto);
proto = proto(class',:);
res.x = (x * w0)';

positive_w0 = w0 > 0;
negtive_w0 = w0 < 0; 
    
% batch_protos = awa_proto(class,:);
loss = 0;
% deal with the w of each column
for i = 1 : size(proto,2)
    mask_abs_dis = ones(size(x,1), 1);
    % |W*j xi - uzij| - epsilon
    abs_dis = x * w0(:, i) - proto(:,i);
    dis_rep = abs(abs_dis) - epsilon;
    
    mask = dis_rep > 0;
    mask_abs_dis(abs_dis < 0) = -1;
    
    diver = (mask .* mask_abs_dis .* dis_rep)' * x * 2;
 
    w0(:, i) = w0(:, i) - param.learningRate * (diver' + param.lambda * (positive_w0(:,i) - negtive_w0(:, i)));
    
    loss = loss + sum((dis_rep).^2) + param.lambda * sum(abs(w0(:,i)));
        
   
end


res.loss = loss;
    

end







