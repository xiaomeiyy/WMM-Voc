function [w0, res] = vl_MMVoc(w0, x, class, param, awa_proto)

param.neighbour_exclude=5;
param.neighbour_nrmd = 5;
param.maximum_margin=7;
% param.epsilon=0.001;
%param.alpha=1;
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
    %res.loss=sum(sum((res.x-proto).^2)) + size(x,2) * sum(sum(abs(w0))) * param.lambda;
    res.loss=sum(sum((res.x-proto).^2));
    
    %%SVR
    positive_w0 = w0 > 0;
    negtive_w0 = w0 < 0;
    g_svr = 2 * ((w0' * x - proto) * x')';
    %g_svr = g_svr / size(x,1);
    %g_svr = g_svr / norm(g_svr,2);
    
    %ssvoc
    g_ssvoc = get_g_ssvoc_v1_w(x',class,param,w0,proto);%include SSVRLossU1-u2 and SSVRLoss_wrt_w  
    
    gw = param.alpha * g_svr + (1-param.alpha) * g_ssvoc  + param.lambda * (positive_w0 - negtive_w0);

    w0=w0-gw*param.learningRate; 

end




function g_ssvoc = get_g_ssvoc_v1_w(feature,class,param,w0,proto)
    
    exclude_proto = param.exclus_proto;
    nrmd_proto = param.nrmd_proto;

    %fix nums
    feature_dim = size(feature,2);
    instance_num = size(feature,1);
    proto_dim = size(w0,2);
    
    %matrix size
    neighbour_num = param.neighbour_exclude + param.neighbour_nrmd;
    extend_num = neighbour_num * instance_num;
    
    l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/proto_dim),1,size(x,2)));
    
    %initial matrix
    u_true = zeros(extend_num,proto_dim);
    u_neighbour = zeros(extend_num, proto_dim);
    extend_x = zeros(extend_num, feature_dim);
    
    %fill data
    cnt = 0;
    for i = 1: instance_num
        %u_single = l2norm(proto(class(i),:));
        u_true((cnt+1):(cnt+neighbour_num),:) = repmat(proto(:,i)', neighbour_num, 1);
        extend_x((cnt+1):(cnt+neighbour_num),:) = repmat(feature(i,:), neighbour_num, 1);
        
        %neighbour exclude proto
        for j = 1 : param.neighbour_exclude
            u_neighbour((cnt+j),:) = l2norm(exclude_proto{1,class(i)}{1,j});
        end
        %neighbour nrm proto
        for k = 1 : param.neighbour_nrmd
            u_neighbour((cnt+param.neighbour_exclude+k),:) = l2norm(nrmd_proto{1,class(i)}{1,k});
        end
        
        cnt = cnt + neighbour_num;       
    end
    
    
    %g_ssvoc = ((w0' * extend_x' - u_true') * extend_x)' - ((w0' * extend_x' - u_neighbour') * extend_x)';
    
    g_ssvoc = zeros(size(w0));
    %%compute gradient
    positive_X2_num = 0 ;
    for i = 1 :extend_num
        xi = extend_x(i,:);
        ui = u_true(i,:);
        ai = u_neighbour(i,:);
        % X2 = C + 0.5 * D(xi,ui) - 0.5 * D(xi, ai)
        X2 = param.maximum_margin + 0.5 * sum((w0' * xi' - ui').^2) - 0.5 * sum((w0' * xi' - ai').^2);
        if X2 > 0
            gi = X2 * (((w0' * xi' - ui') * xi)' - ((w0' * xi' - ai') * xi)');
            positive_X2_num = positive_X2_num + 1;
        else
            gi = 0;
        end
        g_ssvoc = g_ssvoc + gi;      
    end
    if sum(sum(g_ssvoc)) ~= 0
       %g_ssvoc = g_ssvoc / 16 ;
       g_ssvoc = g_ssvoc / positive_X2_num ;
       %g_ssvoc = g_ssvoc / extend_num ;
       %g_ssvoc = g_ssvoc / norm(g_ssvoc,2);
    end
    
    %fu gradient
    V =(single(diag(ones(size(proto,1),1))));
    g_ssvoc_2 = fu_ssvrloss_wxm(w0,extend_x,u_true,u_neighbour,param.maximum_margin,V);
    
end



