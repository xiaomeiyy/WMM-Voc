function [varargout] = est_dis(awa_proto,exclus_proto,portion,open_set)
dbstop if error
    if nargin<3
        portion = 0.5;
    end
    if nargin<4
        open_set = 1;
    end
     dist_param.kappa = zeros(size(awa_proto,1),1);
        dist_param.lambda = zeros(size(awa_proto,1),1);
    if open_set
       
        d = [];
        for i = 1:size(awa_proto,1)
            for j = 1:length(exclus_proto{1})
                d(j)=pdist2(awa_proto(i,:),exclus_proto{i}{j});
            end
    %         d = exp(exp(1./(2-d)-1/2));
            d= d(d>0)*portion;
            l_k = wblfit(d);
            dist_param.lambda(i) = l_k(1);
            dist_param.kappa(i) = l_k(2);
        end
    else
        for i = 1:size(awa_proto,1)
            d = pdist2(awa_proto(i,:),awa_proto([1:(i-1) (i+1):size(awa_proto,1)],:));
            d= d*portion;
            l_k = wblfit(d);
            dist_param.lambda(i) = l_k(1);
            dist_param.kappa(i) = l_k(2);
        end
    end

	varargout{1} =dist_param.kappa;
    varargout{2} =dist_param.lambda;
    
    
	