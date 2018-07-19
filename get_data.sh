if [ -z "$*" ]; 
	then 
		echo "No dataset specified"; 
		echo "Datasets include: snli, squad, wikitext"; 
		echo "Pretrained vectors include: glove, elmo, pretrained_lm"; 
		echo "Try:"; 
		echo "bash get_data.sh 'dataset' 'dataset' ..."; 
		exit 1;
	fi;

PWD=$(pwd);

for ARG in "$@"
	do
		if [ $ARG == "snli" ]; 
		then 
			echo "Downloading snli 1.0"; 
			SNLI_DIR=$PWD/data;
			wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -O $SNLI_DIR/temp.zip; 
			unzip $SNLI_DIR/temp.zip -d $SNLI_DIR/ -x '__MACOSX/*'; 
			rm $SNLI_DIR/temp.zip;
		fi;
done;

for ARG in "$@"
	do
	if [ $ARG == "squad" ]; 
		then 
			echo "Downloading squad 1.1";
			SQUAD_DIR=$PWD/data/squad_1.1;
			mkdir -p $SQUAD_DIR;
			wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json;
			wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json;
		fi;
done;

for ARG in "$@"
	do
	if [ $ARG == "wikitext" ]; 
		then 
			echo "Downloading wikitext 103";
			WIKI_DIR=$PWD/data;
			wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -O $WIKI_DIR/temp.zip; 
			unzip $WIKI_DIR/temp.zip -d $WIKI_DIR/; 
			rm $WIKI_DIR/temp.zip 
		fi;
done;

for ARG in "$@"
	do
	if [ $ARG == "glove" ]; 
		then 
			echo "Downloading glove 300d vectors (matrix of 400k vectors)"; 
			GLOVE_DIR=$PWD/data/pretrained_glove_vectors;
			mkdir -p $GLOVE_DIR;
			wget https://www.dropbox.com/s/q7j80ok91j7y3bc/glove_300_400k_matrix.npy?dl=0 -O $GLOVE_DIR/glove_300_400k_matrix.npy;
		fi;
done;

for ARG in "$@"
	do
	if [ $ARG == "elmo" ]; 
		then 
			echo "Downloading elmo 768d vectors (matrix of 40k vectors)"; 
			ELMO_DIR=$PWD/data/pretrained_elmo_vectors;
			mkdir -p $ELMO_DIR;
			wget https://www.dropbox.com/s/8mzr0fhdwg87ik3/elmo_768_40478_matrix.npy?dl=0 -O $ELMO_DIR/elmo_768_40478_matrix.npy;
		fi;
done;

for ARG in "$@"
	do
	if [ $ARG == "pretrained_lm" ]; 
		then 
			echo "Downloading pretrained language model params"; 
			PRE_LM_DIR=$PWD/data/pretrained_language_model_params;
			mkdir -p $PRE_LM_DIR;
			wget https://www.dropbox.com/s/rsxc6rma9icqswf/pretrained_language_model_params.zip?dl=0 -O $PRE_LM_DIR/temp.zip;
			unzip $PRE_LM_DIR/temp.zip -d $PRE_LM_DIR/; 
			rm $PRE_LM_DIR/temp.zip;
		fi;
done;