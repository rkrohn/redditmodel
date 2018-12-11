#include "stdafx.h"
#include "Snap.h"
#include "word2vec.h"

//Code from https://github.com/nicholas-leonard/word2vec/blob/master/word2vec.c
//Customized for SNAP and node2vec

void LearnVocab(TVVec<TInt, int64>& WalksVV, TIntV& Vocab)
{
	//init all locations to 0
	for( int64 i = 0; i < Vocab.Len(); i++)
		Vocab[i] = 0;

	//loop all walks
	for( int64 i = 0; i < WalksVV.GetXDim(); i++)
	{
		//loop all nodes/words in current walk
		for( int j = 0; j < WalksVV.GetYDim(); j++) 
		{
			Vocab[WalksVV(i,j)]++;		//add to usage counter for this node
		}
	}
}

//Precompute unigram table using alias sampling method
//Vocab is just a frequency table
void InitUnigramTable(TIntV& Vocab, TIntV& KTable, TFltV& UTable)
{
	double TrainWordsPow = 0;
	double Pwr = 0.75;

	//assign probability to each word in vocabulary
	TFltV ProbV(Vocab.Len());
	for (int64 i = 0; i < Vocab.Len(); i++)
	{
		ProbV[i]=TMath::Power(Vocab[i],Pwr);		//prob = word freq ^ 0.75
		TrainWordsPow += ProbV[i];					//running total of all word probabilities
		KTable[i]=0;					//init table entries for thisi word to 0
		UTable[i]=0;
	}
	//finish word probabilities by dividing each by the total prob sum
	for (int64 i = 0; i < ProbV.Len(); i++)
	{
		ProbV[i] /= TrainWordsPow;
	}

	TIntV UnderV;	//list of words with Uvals < 1
	TIntV OverV;	//list of words with Uvals > 1
	//loop all probabilities
	for (int64 i = 0; i < ProbV.Len(); i++)
	{
		UTable[i] = ProbV[i] * ProbV.Len();		//UTable(float) = word prob * size of vocabulary
		//build lists of words split by Uvalue - under or over 1
		if ( UTable[i] < 1 )
		{
			UnderV.Add(i);
		} 
		else
		{
			OverV.Add(i);
		}
	}

	//process words in pairs, one with val under 1 and one with val over 1
	while(UnderV.Len() > 0 && OverV.Len() > 0)
	{
		//get pair of words, remove from processing list
		int64 Small = UnderV.Last();	//small-val word
		int64 Large = OverV.Last();		//large-val word
		UnderV.DelLast();
		OverV.DelLast();

		KTable[Small] = Large;		//set ktable
		UTable[Large] = (UTable[Large] + UTable[Small]) - 1;	//update utable for large-val word
		//add updated u-val word to correct processing list
		if (UTable[Large] < 1)
		{
			UnderV.Add(Large);
		} 
		else
		{
			OverV.Add(Large);
		}
	}

	//process any lingering words with vals < 1
	while(UnderV.Len() > 0)
	{
		//get current word from processing list and remove
		int64 curr = UnderV.Last();
		UnderV.DelLast();
		UTable[curr] = 1;		//set uval to 1
	}
	//process any lingering words with vals > 1
	while(OverV.Len() > 0)
	{
		//get word, remove from list
		int64 curr = OverV.Last();
		OverV.DelLast();
		UTable[curr] = 1;		//set uval to 1
	}

	/*
	printf("Unigram Table\n");
	for (int64 i = 0; i < UTable.Len(); i++)
		printf("%lf  ", UTable[i]);
	printf("\n");

	printf("K Table\n");
	for (int64 i = 0; i < KTable.Len(); i++)
		printf("%lf  ", KTable[i]);
	printf("\n");
	*/
}

//draw a random word based on unigram table for negative sampling (math magic)
int64 RndUnigramInt(TIntV& KTable, TFltV& UTable, TRnd& Rnd)
{
	TInt X = KTable[static_cast<int64>(Rnd.GetUniDev()*KTable.Len())];	//random int index for unigram/ktable
	double Y = Rnd.GetUniDev();		//random float
	return Y < UTable[X] ? X : KTable[X];
}

//Initialize negative embeddings - set all to 0
//this is the output layer, will get thrown out at the end
void InitNegEmb(TIntV& Vocab, const int& Dimensions, TVVec<TFlt, int64>& SynNeg)
{
	SynNeg = TVVec<TFlt, int64>(Vocab.Len(),Dimensions);
	for (int64 i = 0; i < SynNeg.GetXDim(); i++)	//loop nodes/words
	{
		for (int j = 0; j < SynNeg.GetYDim(); j++) 		//loop embedding dimensions
		{
			SynNeg(i,j) = 0;		//set all to 0
		}
	}
}

//Initialize positive embeddings - set all to random values
//this is the hidden layer, aka the final embeddings
void InitPosEmb(TIntV& Vocab, const int& Dimensions, TRnd& Rnd, TVVec<TFlt, int64>& SynPos, TIntFltVH& InitEmbeddingsHV, TIntIntH& RnmBackH)
{
	//printf("Initializing embeddings...\n");
	SynPos = TVVec<TFlt, int64>(Vocab.Len(),Dimensions);

	//set up default embedding values
	TFltV DefaultEmbeddingV;	//[1, 2, 0.75], [0.15, 1.5], 0.05
	DefaultEmbeddingV[0] = 1.0;
	DefaultEmbeddingV[1] = 2.0;
	DefaultEmbeddingV[2] = 0.75;
	DefaultEmbeddingV[3] = 0.15;
	DefaultEmbeddingV[4] = 1.5;
	DefaultEmbeddingV[5] = 0.05;

	for (int64 i = 0; i < SynPos.GetXDim(); i++)	//loop nodes/words
	{
		//fetch initial embeddings vector for this word
		int64 orig_id = RnmBackH.GetDat(i);		//get original id
		//printf("%d -> %d: ", orig_id, i);
		bool valid = false;
		TFltV CurrV;
		if (InitEmbeddingsHV.IsKey(orig_id))
		{
			valid = true;
			CurrV = InitEmbeddingsHV.GetDat(orig_id);
			//printf("copy \n");
		}		
		for (int j = 0; j < SynPos.GetYDim(); j++) 		//loop embedding dimensions
		{
			if (valid)
			{
				SynPos(i,j) = CurrV[j];		//if have initial embedding, use it
			}
			else
				SynPos(i,j) = DefaultEmbeddingV[j] + 2*(Rnd.GetUniDev()-0.5)*(0.15*DefaultEmbeddingV[j]);		
				//random values, ranging to 0/dimensions to 1/dimensions (all positive)
				//new version: default hardcoded based on param defaults from fitting, +/-15% random delta
			//printf("%f ", SynPos(i, j));
		}
		//printf("\n");
	}
}

//iterative training? called iter * #walks times
void TrainModel(TVVec<TInt, int64>& WalksVV, const int& Dimensions,
		const int& WinSize, const int& Iter, const bool& Verbose,
		TIntV& KTable, TFltV& UTable, int64& WordCntAll, TFltV& ExpTable,
		double& Alpha, int64 CurrWalk, TRnd& Rnd,
		TVVec<TFlt, int64>& SynNeg, TVVec<TFlt, int64>& SynPos, TIntFltH& StickyFactorsH, TIntIntH& RnmBackH)
		{
	//neuron layers: one is internal embeddings, one is the output later, but I'm not sure which is which yet
	//HAHA! one of them isn't even used - kill it with fire!
	//TFltV Neu1V(Dimensions);
	TFltV Neu1eV(Dimensions);

	int64 AllWords = WalksVV.GetXDim()*WalksVV.GetYDim();	//number of walks * length of walks

	//extract current walk into it's own vector
	TIntV WalkV(WalksVV.GetYDim());
	//printf("\nCURRENT WALK: ");
	for (int j = 0; j < WalksVV.GetYDim(); j++) 
	{ 
		WalkV[j] = WalksVV(CurrWalk,j); 
		//printf("%d ", WalkV[j]);
	}
	//printf(" (len %d)", WalkV.Len());

	//loop nodes/words in current walk
	for (int64 WordI=0; WordI<WalkV.Len(); WordI++)
	{
		//printf("\nProcessing position %d in current walk (walk %d)\n", WordI, CurrWalk);

		//every 10,000 words, do some things
		if ( WordCntAll%10000 == 0 ) 
		{
			if ( Verbose )		//period prints (progress bar)
			{
				printf("\rLearning Progress: %.2lf%% ",(double)WordCntAll*100/(double)(Iter*AllWords));
				fflush(stdout);
			}
			//update alpha
			Alpha = StartAlpha * (1 - WordCntAll / static_cast<double>(Iter * AllWords + 1));
			if ( Alpha < StartAlpha * 0.0001 ) { Alpha = StartAlpha * 0.0001; }
		}

		int64 Word = WalkV[WordI];			//pull current word from walk

		//init neuron vectors to 0 across all dimensions
		for (int i = 0; i < Dimensions; i++)
		{
			//Neu1V[i] = 0;
			Neu1eV[i] = 0;
		}

		int Offset = Rnd.GetUniDevInt() % WinSize;		//draw random offset, from 0 to WinSize-1
														//this is the amount we will "shrink" the window size by
		//printf("   Offset %d : ", Offset);
		//a is the offset into the current window, relative to the window start
		//window start is defined by the randomly-drawn Offset - which defines how many words at the start of the window to skip
		//(a is NOT a walk/sentence index - relative to window)
		for (int a = Offset; a < WinSize * 2 + 1 - Offset; a++)
		{
			if (a == WinSize) { continue; }		//if a equals context size, would process current word as neighbor, skip

			//convert window index to walk index
			int64 CurrWordI = WordI - WinSize + a;	//current pos = current walk position - context size + a
			//if this position beyond walk, skip
			if (CurrWordI < 0){ continue; }
			if (CurrWordI >= WalkV.Len()){ continue; }

			//printf("%d ", a);

			int64 CurrWord = WalkV[CurrWordI];		//pull word from walk corresponding to this position
			for (int i = 0; i < Dimensions; i++) { Neu1eV[i] = 0; }		//reset neuron

			//negative sampling: each training sample only modifies a small percentage of the weights, instead of all of them
			//this sample of words is selected using the unigram distribution previously computed)
			//NegSamN = number of words to update
			//add one to NegSamN because use one iteration to train the positive sample (current center word)
			for (int j = 0; j < NegSamN+1; j++)		
			{
				int64 Target, Label;
				//first loop: set target equal to current word, label to 1
				//train the positive sample, always
				if (j == 0)
				{
					Target = Word;
					Label = 1;		//label is 1 in output layer
				} 
				//other iterations, train a negative sample
				else
				{
					Target = RndUnigramInt(KTable, UTable, Rnd);	//draw random word
					if (Target == Word) { continue; }	//if drew the current walk word, next iteration - don't use as negative sample!
					Label = 0;		//target not same as current word, label for this word is 0 (negative sample)
				}

				//Target is the word to update the model for
				//Label indicates whether Target is a positive (1) or negative (0) example

				//for current neighbor word, multiply their synpos * synneg, and sum all products
				//compute dot-product between input word weights (SynPos) and output word weights (SynNeg)
				double Product = 0;
				for (int i = 0; i < Dimensions; i++)
				{
					Product += SynPos(CurrWord,i) * SynNeg(Target,i);
				}

				//compute the error at the output, store in Grad
				//subtract network output from desired output, and multiply by learning rate
				double Grad;                     //Gradient multiplied by learning rate
				if (Product > MaxExp) { Grad = (Label - 1) * Alpha; }
				else if (Product < -MaxExp) { Grad = Label * Alpha; }
				else
				{
					double Exp = ExpTable[static_cast<int>(Product*ExpTablePrecision)+TableSize/2];
					Grad = (Label - 1 + 1 / (1 + Exp)) * Alpha;
				}

				//update vectors based on gradient (gradient calculation?)
				for (int i = 0; i < Dimensions; i++)
				{
					//multiply error by output layer weights, accumulate over all negative samples
					Neu1eV[i] += Grad * SynNeg(Target,i);
					//update outer layer weights: multiply output error by hidden layer weights	
					SynNeg(Target,i) += Grad * SynPos(CurrWord,i);	
				}
			}

			//finished negative sampling for current neighbor word, update it's synpos
			//updates hidden weights after hidden layer gradients for all negative samples have been accumulated
			//does this word/node have a sticky factor? if so, use it
			//fetch initial embeddings vector for this word
			int64 orig_id = RnmBackH.GetDat(CurrWord);		//get original id for CurrWord
			TFlt CurrSticky;
			if (StickyFactorsH.IsKey(orig_id))
			{
				CurrSticky = StickyFactorsH.GetDat(orig_id);		//use cached sticky factor
				//printf("%d -> %d: %f\n", orig_id, CurrWord, CurrSticky);
			}
			else
			{
				CurrSticky = 3.0;		//no sticky provided, use 1 for full adjustment effect
				//printf("%d -> %d: no sticky, use 1.0\n", orig_id, CurrWord);
			}
			//update hidden weight
			for (int i = 0; i < Dimensions; i++)
			{
				if (SynPos(CurrWord,i) + CurrSticky * Neu1eV[i] > 0)
					SynPos(CurrWord,i) += CurrSticky * Neu1eV[i];		//this is where the embedding gets updated
																		//but only update if result is non-negative
			}
		}
		WordCntAll++;		//finished current word in walk, update counter
	}
}

//learn embeddings for "words" (nodes) based on "sentences" (walks)
//pass in: walks, embedding dimensions/size, context window size, number of SGD epochs, verbose flag, container for embeddings
//note that the walks may contain a 0, which indicates no node travel there - why it works that way, I do not know
void LearnEmbeddings(TVVec<TInt, int64>& WalksVV, const int& Dimensions,
	const int& WinSize, const int& Iter, const bool& Verbose,
	TIntFltVH& EmbeddingsHV, TIntFltVH& InitEmbeddingsHV, TIntFltH& StickyFactorsH)
	{
	TIntIntH RnmH;		//hash type for mapping, given node id -> consecutive node id
	TIntIntH RnmBackH;	//reverse hash, assigned consecutive node id -> given node id
	int64 NNodes = 0;	//node counter

	//renaming nodes into consecutive numbers (because reasons)
	for (int i = 0; i < WalksVV.GetXDim(); i++)
	{
		for (int64 j = 0; j < WalksVV.GetYDim(); j++) 
		{
			//if nodeid at i,j is already key, reset to that key
			if ( RnmH.IsKey(WalksVV(i, j)) )
			{
				WalksVV(i, j) = RnmH.GetDat(WalksVV(i, j));
			} 
			//if not key, add to both hashes
			else
			{
				RnmH.AddDat(WalksVV(i,j),NNodes);
				RnmBackH.AddDat(NNodes,WalksVV(i, j));
				WalksVV(i, j) = NNodes++;		//reset to newly assigned hash key, move to next
			}
		}
	}

	//learn vocabulary from random walks
	TIntV Vocab(NNodes);		//arry-type container, one space for each "word" (node), holds frequencies
	LearnVocab(WalksVV, Vocab);		//count number of times each word/node is used

	
	//printf("Vocab (%d words)\n", Vocab.Len());
	/*
	for (int64 i = 0; i < Vocab.Len(); i++)
		printf("%.10e  ", Vocab[i]);
	printf("\n");
	*/

	TIntV KTable(NNodes);		//vector of integers, length of vocabulary
	TFltV UTable(NNodes);		//vector of floats, lenght of vocabulary
	//embedding vectors, both type fload
	TVVec<TFlt, int64> SynNeg;
	TVVec<TFlt, int64> SynPos;	//this is actually the embedding returned at the end of training
	TRnd Rnd(time(NULL));		//seed the randomizer

	//initialize positive and negative embeddings: len(vocab) * Dimensions
	InitPosEmb(Vocab, Dimensions, Rnd, SynPos, InitEmbeddingsHV, RnmBackH);		//init embedding with random values - but not random anymore!
	InitNegEmb(Vocab, Dimensions, SynNeg);			//all 0

	
	//printf("SynPos dim: (%d, %d)\n", SynPos.GetXDim(), SynPos.GetYDim());
	/*
	printf("SynPos (first rows only)\n");
	for (int64 i = 0; i < SynPos.GetXDim(); i++)	//loop nodes/words
	{
		for (int j = 0; j < SynPos.GetYDim(); j++) 		//loop embedding dimensions
		{
			printf("%lf ", SynPos(i,j));
		}
		printf("\n");
		if (i == 0)
			break;
	}
	*/

	InitUnigramTable(Vocab, KTable, UTable);		//compute unigram table - somewhat related to prob/freq?

	//build exptable
	//printf("Table Size: %d\n", TableSize);
	TFltV ExpTable(TableSize);
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < TableSize; i++ )
	{
		double Value = -MaxExp + static_cast<double>(i) / static_cast<double>(ExpTablePrecision);
		ExpTable[i] = TMath::Power(TMath::E, Value);
	}

	int64 WordCntAll = 0;
	double Alpha = StartAlpha;                              //learning rate
// op RS 2016/09/26, collapse does not compile on Mac OS X
//#pragma omp parallel for schedule(dynamic) collapse(2)
	//loop number of SGD iterations
	for (int j = 0; j < Iter; j++)
	{
//#pragma omp parallel for schedule(dynamic)		PUT THIS BACK!!!
		//loop walks
		for (int64 i = 0; i < WalksVV.GetXDim(); i++)
		{
			//for each walk, train the model
			//arguments: vector of walks, embedding dimensions, context size, SGD iterations, verbose flag
			//ktable, unigram table, wordcount, exptable, learning rate alpha, randomizer, 
			//synneg (all 0, one per embedding pos), synpos (random values, one per embedding val)
			TrainModel(WalksVV, Dimensions, WinSize, Iter, Verbose, KTable, UTable,
			 WordCntAll, ExpTable, Alpha, i, Rnd, SynNeg, SynPos, StickyFactorsH, RnmBackH); 
			//this updates embeddings based on the current walk given to the function
		}
	}
	if (Verbose) { printf("\n"); fflush(stdout); }

	//loop node rows
	for (int64 i = 0; i < SynPos.GetXDim(); i++)
	{
		//copy synpos 
		TFltV CurrV(SynPos.GetYDim());
		//loop embedding dimensions
		for (int j = 0; j < SynPos.GetYDim(); j++) 
		{ 
			CurrV[j] = SynPos(i, j); 	//copy value
		}
		EmbeddingsHV.AddDat(RnmBackH.GetDat(i), CurrV);	//add this word's embeddings to overall
	}
}
