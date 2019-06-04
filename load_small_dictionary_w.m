%% load_small_dictionary: load small dictionary
function [dictionary,vocab] = load_small_dictionary_w(data_path)
	if nargin<1
		data_path = '/home/fu/Desktop/experiments_wxm/ZSL_ssvoc/datasets/data/dictionary';
	end
	load([data_path,'/selected_smaller_dic.mat']);
	vocab = selected_vocab;
	dictionary = selected_dict;





	
