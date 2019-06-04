%% load_exclus_proto
function [exclus_proto] = load_exclus_proto_w(prototypes, dictionary, ntop,y)
	if nargin<3
		ntop = 50;
    end
    if nargin<4
        dis = pdist2(prototypes,dictionary,'cosine');
        [~,idx] = sort(dis,2,'ascend');
        exclus_proto_idx = idx(:,2:ntop);
        for i=1:size(exclus_proto_idx,1)
            for j=1:size(exclus_proto_idx,2) 
               exclus_proto{i}{j} = dictionary(exclus_proto_idx(i,j),:);
            end    
        end
    else
        dis = pdist2(prototypes,dictionary,'cosine');
        uni_y = unique(y);
        for i = 1:length(uni_y)
            idx_y = find(y==uni_y(i));
            dis(idx_y,idx_y)=100;
        end           
        [~,idx] = sort(dis,2,'ascend');
        exclus_proto_idx = idx(:,1:ntop);
        for i=1:size(exclus_proto_idx,1)
            for j=1:size(exclus_proto_idx,2) 
               exclus_proto{i}{j} = dictionary(exclus_proto_idx(i,j),:);
            end    
        end
    end
        
