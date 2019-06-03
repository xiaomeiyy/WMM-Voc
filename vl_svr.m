function [w0, res] = vl_awa2_v2(w0, x, class, param, awa_proto)

% param.neighbour=5;
% param.neighbour_prototype = 5;
% param.maximum_margin=7;
% param.epsilon=0.001;
param.alpha=1;
%param.learningrate=0.000001;
% param.momentum=0.9;
% param.VERSION=0;
param.lambda = 0.01;

kdim = size(awa_proto,2);
l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/kdim),1,size(x,2)));%球形范数�?
proto = l2norm(awa_proto);
proto = proto(class',:);
    
    
    
    x = x'; %4096 * n
    proto = proto'; %100*n
    res.x=w0'* x;
    res.loss=sum(sum((res.x-proto).^2)) + size(x,2) * sum(sum(abs(w0))) * param.lambda;
    
    positive_w0 = w0 > 0;
    negtive_w0 = w0 < 0;
    g_svr = 2 * ((w0' * x - proto) * x')' + param.lambda * (positive_w0 - negtive_w0);
    gw = param.alpha * g_svr;

    w0=w0-gw*param.learningRate; 

end

