function [w0,state] = awa_resnet_deep_MMVoc_main()
clear;clc;
awaopt = struct([]);
awaopt =getPrmDflt(awaopt,{'updating_v',false,'openset',false,...
        'updating_k_l',true,'portion',0.3,...
        'ini_w',false,...
        'load_dict',@load_small_dictionary_w,...
        'load_data',@load_awa_resnet_w_split,...
        'load_proto',@load_awa_proto,...
        'load_exclude',@load_exclus_proto_w,...
        'pca',false,'pca_dim',100,'exclus_ntop',5,...
        'usingGPU',0,'feat2sem',1},-1);
% param.alpha = 0.1;   
% param.maximum_margin = 7;
% param.VERSION=1; 
% param.neighbour=5; 
% param.neighbour_protoype = 5;
% param.svr_loss = 1;
% param.strongly=0;
% param.useBatchingLBFGS=0;
% param.maxIter_w =20;

%% LOAD DATA
% [x,y,awa_word] = awaopt.load_data();
[dictionary,vocab] = awaopt.load_dict();
% [awa_proto] = awaopt.load_proto(dictionary,vocab,awa_word);
%[vocab,dictionary] = add_prototype(vocab,dictionary,awa_word,awa_proto);
load('awa_resnet_datasets.mat');
load('awa_proto_100.mat');

opts.exclus_proto=awaopt.load_exclude(awa_proto,dictionary);
opts.nrmd_proto=awaopt.load_exclude(awa_proto,awa_proto);

% calculate the paras of weibull distribution
[kappa,lambda]= est_dis(awa_proto,opts.exclus_proto,awaopt.portion);
opts.dis_kappa = kappa;
opts.dis_lambda = lambda;

%test vl_fu_ssvoc
meta_proto.exclus_proto = opts.exclus_proto;
kdim = size(awa_proto,2);
l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/kdim),1,size(x,2)));
nrmawa_prot = l2norm(awa_proto);
dis=pdist2(nrmawa_prot,nrmawa_prot,'cosine');%自己和自己的距离？
[val,idx]=sort(dis,2,'ascend');
cls_idx = idx(:,2:5+1);
meta_proto.nrm_proto = nrmawa_prot;
meta_proto.cls_idx = cls_idx;
opts.meta_proto = meta_proto;

%five instances each class
[x,y] = num5instance(x,y,5);


%% SGD
opts.expDir = fullfile('data') ;
opts.continue = true ;
opts.plotStatistics = true;
opts.batchSize = 16;
opts.learningRates = 0.00004 * [0.93*ones(1,200) 0.25*ones(1,5) 0.12*ones(1,5)] ;
%opts.alpha = [0.9*ones(1,3) 0.3 0.4 0.5 0.6*ones(1,200) 0.12*ones(1,5)] ;
opts.alpha = [0.6*ones(1,150)] ;
opts.numEpochs = numel(opts.learningRates) ;
opts.train = [];
opts.numSubBatches = 1 ;
opts.errorLabels = {'top1err', 'top5err'} ;

fea_dim = size(x.tr,2);
proto_dim = size(awa_proto,2);
w0 = single(normrnd(0,0.0001,fea_dim,proto_dim));

if isempty(opts.train), opts.train = find(y.tr > 0) ; end
evaluateMode = isempty(opts.train) ;
modelPath = @(ep) fullfile(opts.expDir, sprintf('w0-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'w0-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [w0, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

for epoch = start + 1 : opts.numEpochs
    params = opts;
    params.epoch = epoch;
    params.learningRate = opts.learningRates(min(epoch,numel(opts.learningRates)));
    params.alpha = opts.alpha(min(epoch,numel(opts.alpha)));
    params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    params.imdb.x = x;
    params.imdb.y = y;
    params.getBatch = @getSimpleNNBatchW;
    params.errorFunction = @ssvoc_error_w;
    
    [w0, state] = processEpochTrain(w0,state,params,awa_proto);
    
    if ~evaluateMode
      saveStateW(modelPath(epoch), w0, state) ;
    end
    lastStats = state.stats ;   
    
    % validation
    lastStats.val = zls_err(w0,x.te,y.te,awa_proto);
    
    stats.train(epoch) = lastStats.train ;
    stats.val(epoch) = lastStats.val;
    clear lastStats ;
    if ~evaluateMode
        saveStatsW(modelPath(epoch), stats) ;
    end

    if params.plotStatistics
      switchFigure(1) ; clf ;
      %plots = setdiff(cat(2,fieldnames(stats.train)',fieldnames(stats.val)'), {'num', 'time'}) ;
      plots = setdiff(cat(2,fieldnames(stats.train)'), {'num', 'time'}) ;
      for p = plots
        p = char(p) ;
        values = zeros(0, epoch) ;
        leg = {} ;
        for f = {'train','val'}
          f = char(f) ;
          if isfield(stats.(f), p)
            tmp = [stats.(f).(p)] ;
            values(end+1,:) = tmp(1,:)' ;
            leg{end+1} = f ;
          end
        end
        subplot(1,numel(plots),find(strcmp(p,plots))) ;
        plot(1:epoch, values','o-') ;
        xlabel('epoch') ;
        title(p) ;
        legend(leg{:}) ;
        grid on ;
      end
      drawnow ;
      print(1, modelFigPath, '-dpdf') ;
    end
  
end

end


% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
    if get(0,'CurrentFigure') ~= n
      try
        set(0,'CurrentFigure',n) ;
      catch
        figure(n) ;
      end
    end
end

% -------------------------------------------------------------------------
function saveStatsW(fileName, stats)
% -------------------------------------------------------------------------
    if exist(fileName)
      save(fileName, 'stats', '-append') ;
    else
      save(fileName, 'stats') ;
    end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
    list = dir(fullfile(modelDir, 'w0-*.mat')) ;
    tokens = regexp({list.name}, 'w0-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
end


% -------------------------------------------------------------------------
function saveStateW(fileName, w0, state)
% -------------------------------------------------------------------------
save(fileName, 'w0', 'state') ;
end

% -------------------------------------------------------------------------
function [w0, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'w0', 'state', 'stats') ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end
end

% -------------------------------------------------------------------------
function [w0, state] = processEpochTrain(w0,state,params,awa_proto)
% -------------------------------------------------------------------------
mode = 'train';
subset = params.(mode);
num = 0 ;
error = [] ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:params.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    [im, labels] = params.getBatch(params.imdb, batch) ;
   
    %[w0, res] = vl_ssvoc_fu(w0, im, labels, params, awa_proto) ;
    [w0, res] = vl_MMVoc(w0, im, labels, params, awa_proto) ;
    %[w0, res] = vl_ssvoc_fu_v2(w0, im, labels, params, awa_proto) ;

    %accumulate errors
    error = sum([error, [sum(double(res.loss));reshape(params.errorFunction(res, labels, awa_proto),[],1);]],2);
  end
  
   % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = extractStatsW(params, error / num) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s: %.3f', f, stats.(f)) ;
  end
  fprintf('\n') ;

end

% Save back to state.
state.stats.(mode) = stats ;

end

% -------------------------------------------------------------------------
function stats = extractStatsW(params, errors)
% -------------------------------------------------------------------------
stats.loss = errors(1);
for i = 1:numel(params.errorLabels)
  stats.(params.errorLabels{i}) = gather(errors(i+1)) ;
end
end

% -------------------------------------------------------------------------
function val = zls_err(w0,fea,labels,awa_proto)
% -------------------------------------------------------------------------
    %prepare class prototype and output x
    kdim = size(awa_proto,2);
    l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/kdim),1,size(x,2)));
    wt_label = unique(labels);
    proto = single(l2norm(awa_proto(wt_label,:)));
    inst_labels = zeros(1,numel(labels));
    for i = 1 : numel(labels)
        inst_labels(i) = find(labels(i) == wt_label);
    end

    x = w0' * fea'; 

    %calculate distance between every class of each instance
    predictions = pdist2(proto,x','cosine');

    [vals,predictions] = sort(predictions, 1, 'ascend') ;

    % valid labels
    mass = single(inst_labels > 0) ;

    m = min(5, size(predictions,1)) ;

    % number of error instances in top1 and top5
    error = ~bsxfun(@eq, predictions, inst_labels) ;
    err(1,1) = sum(mass .* error(1,:));
    err(2,1) = sum(mass .* min(error(1:m,:),[],1));
    err(3,1) = sum(mass .* min(error(1:2,:),[],1));
    
    val.top1err = err(1,1)/size(fea,1);
    val.top5err = err(2,1)/size(fea,1);
    proto_ws = single(l2norm(awa_proto(labels,:)));
    val.loss = sum(sum((x - proto_ws').^2))/size(fea,1);
end

% -------------------------------------------------------------------------
function err = ssvoc_error_w(res, labels, awa_proto)
% -------------------------------------------------------------------------
%prepare class prototype and output x
kdim = size(awa_proto,2);
l2norm =@(x)(x./repmat(sqrt(sum(x.*x,2)/kdim),1,size(x,2)));
proto = single(l2norm(awa_proto));
x = res.x'; 

%calculate distance between every class of each instance
predictions = pdist2(proto,x,'cosine');

[vals,predictions] = sort(predictions, 1, 'ascend') ;

% valid labels
mass = single(labels > 0) ;

m = min(5, size(predictions,1)) ;

% number of error instances in top1 and top5
error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(mass .* error(1,:));
err(2,1) = sum(mass .* min(error(1:m,:),[],1));
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatchW(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.x.tr(batch,:) ;
labels = imdb.y.tr(1,batch) ;
end