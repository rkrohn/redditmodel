#include "stdafx.h"

#include "n2v.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& OutFile,
	int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter,
	bool& Verbose, double& ParamP, double& ParamQ, bool& Directed, bool& Weighted,
	bool& OutputWalks, TStr& InitInFile, bool& Sticky, bool& OTF) 
{
	Env = TEnv(argc, argv, TNotify::StdNotify);
	Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
	InFile = Env.GetIfArgPrefixStr("-i:", "graph/karate.edgelist",
	"Input graph path");
	InitInFile = Env.GetIfArgPrefixStr("-ie:", "", "Initial embeddings input file. Default is None.");
	OutFile = Env.GetIfArgPrefixStr("-o:", "emb/karate.emb",
	"Output graph path");
	Dimensions = Env.GetIfArgPrefixInt("-d:", 128,
	"Number of dimensions. Default is 128");
	WalkLen = Env.GetIfArgPrefixInt("-l:", 80,
	"Length of walk per source. Default is 80");
	NumWalks = Env.GetIfArgPrefixInt("-r:", 10,
	"Number of walks per source. Default is 10");
	WinSize = Env.GetIfArgPrefixInt("-k:", 10,
	"Context size for optimization. Default is 10");
	Iter = Env.GetIfArgPrefixInt("-e:", 1,
	"Number of epochs in SGD. Default is 1");
	ParamP = Env.GetIfArgPrefixFlt("-p:", 1,
	"Return hyperparameter. Default is 1");
	ParamQ = Env.GetIfArgPrefixFlt("-q:", 1,
	"Inout hyperparameter. Default is 1");
	Verbose = Env.IsArgStr("-v", "Verbose output.");
	OTF = Env.IsArgStr("-otf", "Computing transition probabilities on-the-fly.");
	Directed = Env.IsArgStr("-dr", "Graph is directed.");
	Weighted = Env.IsArgStr("-w", "Graph is weighted.");
	Sticky = Env.IsArgStr("-s", "Using \"sticky\" factor.");
	OutputWalks = Env.IsArgStr("-ow", "Output random walks instead of embeddings.");
}

//read edgelist graph from input file and construct graph
void ReadGraph(TStr& InFile, bool& Directed, bool& Weighted, bool& Verbose, PWNet& InNet) 
{
	TFIn FIn(InFile);
	int64 LineCnt = 0;
	try 
	{
		while (!FIn.Eof()) 
		{
			TStr Ln;
			FIn.GetNextLn(Ln);
			TStr Line, Comment;
			Ln.SplitOnCh(Line,'#',Comment);
			TStrV Tokens;
			Line.SplitOnWs(Tokens);

			//isolated nodes show up in edgelist as a single node id, alone on the line
			if(Tokens.Len()==1)
			{
				int64 NId = Tokens[0].GetInt();
				if (!InNet->IsNode(NId)){ InNet->AddNode(NId); }
			}

			if(Tokens.Len()<2){ continue; }
			int64 SrcNId = Tokens[0].GetInt();
			int64 DstNId = Tokens[1].GetInt();
			double Weight = 1.0;
			if (Weighted) { Weight = Tokens[2].GetFlt(); }
			if (!InNet->IsNode(SrcNId)){ InNet->AddNode(SrcNId); }
			if (!InNet->IsNode(DstNId)){ InNet->AddNode(DstNId); }
			InNet->AddEdge(SrcNId,DstNId,Weight);
			if (!Directed){ InNet->AddEdge(DstNId,SrcNId,Weight); }
			LineCnt++;
		}
		if (Verbose) { printf("Read %lld graph lines from %s\n", (long long)LineCnt, InFile.CStr()); }
	} 
	catch (PExcept Except) 
	{
		if (Verbose) 
		{
			printf("Read %lld graph lines from %s, then %s\n", (long long)LineCnt, InFile.CStr(),
			Except->GetStr().CStr());
		}
	}
}

//read initial embeddings values from file, save to hash
//assumes the sticky factors given are quality, so flip them (1-val) to get true sticky factor
void ReadInitialEmbeddings(TStr& InitInFile, TIntFltVH& InitEmbeddingsHV, bool& Sticky, TIntFltH& StickyFactorsH, bool& Verbose, int Dimensions)
{
	TFIn FIn(InitInFile);
	int64 LineCnt = 0;
	try 
	{
		while (!FIn.Eof()) 
		{
			//get next line of file
			TStr Ln;
			FIn.GetNextLn(Ln);

			//split out comments
			TStr Line, Comment;
			Ln.SplitOnCh(Line,'#',Comment);

			//tokenize the line
			TStrV Tokens;
			Line.SplitOnWs(Tokens);

			//too few tokens, skip this line
			if(Tokens.Len() < Dimensions+1){ continue; }

			//extract tokens
			int64 NId = Tokens[0].GetInt();		//node id
			//printf("%ld ", NId);

			//params for this node
			TFltV CurrV(Dimensions);
			for (int i = 0; i < Dimensions; i++)
			{
				//get embedding value
				CurrV[i] = Tokens[i+1].GetFlt();
				//printf("%f ", CurrV[i]);
				
			}
			InitEmbeddingsHV.AddDat(NId, CurrV);	//add vector to this node's initial embeddings

			//sticky factor if we have it
			if (Sticky && Tokens.Len() >= Dimensions+2)
			{
				TFlt CurrStick = 1 - Tokens[Dimensions+1].GetFlt();
				//printf("(%f)", CurrStick);
				StickyFactorsH.AddDat(NId, CurrStick);
			}

			//printf("\n");
			LineCnt++;
		}
		if (Verbose) { printf("Read %lld param lines from %s\n", (long long)LineCnt, InitInFile.CStr()); }
	} 
	catch (PExcept Except) 
	{
		if (Verbose) 
		{
			printf("Read %lld param lines from %s, then %s\n", (long long)LineCnt, InitInFile.CStr(),
			Except->GetStr().CStr());
		}
	}

	
}

//dump embeddings (and optional walks) to output file
void WriteOutput(TStr& OutFile, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV,
 bool& OutputWalks) 
	{
	TFOut FOut(OutFile);
	if (OutputWalks) 
	{
		for (int64 i = 0; i < WalksVV.GetXDim(); i++) 
		{
			for (int64 j = 0; j < WalksVV.GetYDim(); j++) 
			{
				FOut.PutInt(WalksVV(i,j));
				if(j+1==WalksVV.GetYDim()) 
				{
					FOut.PutLn();
				} 
				else 
				{
					FOut.PutCh(' ');
				}
			}
		}
		return;
	}
	bool First = 1;
	for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) 
	{
		if (First) 
		{
			FOut.PutInt(EmbeddingsHV.Len());
			FOut.PutCh(' ');
			FOut.PutInt(EmbeddingsHV[i].Len());
			FOut.PutLn();
			First = 0;
		}
		FOut.PutInt(EmbeddingsHV.GetKey(i));
		for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) 
		{
			FOut.PutCh(' ');
			FOut.PutFlt(EmbeddingsHV[i][j]);
		}
		FOut.PutLn();
	}
}

int main(int argc, char* argv[])
{
	TStr InFile, InitInFile, OutFile;
	int Dimensions, WalkLen, NumWalks, WinSize, Iter;
	double ParamP, ParamQ;
	bool Directed, Weighted, Verbose, OutputWalks, Sticky, OTF;

	//parse command line args
	ParseArgs(argc, argv, InFile, OutFile, Dimensions, WalkLen, NumWalks, WinSize,
	Iter, Verbose, ParamP, ParamQ, Directed, Weighted, OutputWalks, InitInFile, Sticky, OTF);

	//for now, require the initial embeddings file - because that's what we're doing, and I don't care enough to do this clean right now
	if (InitInFile.Len() == 0)
	{
		printf("Must provide initial embeddings file with -ie option. Exiting.");
		return 0;
	}

	PWNet InNet = PWNet::New();		//network object
	TIntFltVH EmbeddingsHV;			//embeddings object - hash int to vector of floats
	TVVec <TInt, int64> WalksVV;	//walks?
	TIntFltVH InitEmbeddingsHV;		//initial embedding object setting: hash int to vector of floats
	TIntFltH StickyFactorsH;		//sticky factors: hash int node id to float

	ReadGraph(InFile, Directed, Weighted, Verbose, InNet);		//read graph from edgelist

	//read initial embeddings
	ReadInitialEmbeddings(InitInFile, InitEmbeddingsHV, Sticky, StickyFactorsH, Verbose, Dimensions);

	//run node2vec: network, configuration parameters, objects for walks and embeddings
	node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, 
	Verbose, OutputWalks, WalksVV, EmbeddingsHV, InitEmbeddingsHV, StickyFactorsH, OTF);

	//dump results
	WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks);
	return 0;
}
