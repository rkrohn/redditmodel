#include "stdafx.h"
#include "Snap.h"
#include "biasedrandomwalk.h"

//Preprocess alias sampling method
//https://en.wikipedia.org/wiki/Alias_method
//params are normalized transition prob table and CurrI's NI data of type TPair<TVec<TInt, int>, TVec<TFlt, int>>
void GetNodeAlias(TFltV& PTblV, TIntVFltVPr& NTTable)
{
	int64 N = PTblV.Len();		//number of entries in probability table
	TIntV& KTbl = NTTable.Val1;		//integer table - alias table K for alias sampling
	TFltV& UTbl = NTTable.Val2;		//float table - probability table U for alias sampling
	//PTblV serves as the probability distribution p for alias sampling method

	//set both tables to all 0
	for (int64 i = 0; i < N; i++)
	{
		KTbl[i]=0;
		UTbl[i]=0;
	}

	//generate U and K tables for alias sampling method
	TIntV UnderV;	//underfull group
	TIntV OverV;	//overfull group
	for (int64 i = 0; i < N; i++)		//loop number of entries in prob table
	{
		UTbl[i] = PTblV[i]*N;		//U = p * n

		if (UTbl[i] < 1)
		{
			UnderV.Add(i);		//i is underfull
		} 
		else
		{
			OverV.Add(i);		//i is overfull
		}
	}
	//repeat until tables are happy
	while (UnderV.Len() > 0 && OverV.Len() > 0)
	{
		//arbitrary small (underfull) and large (overfull) entries
		int64 Small = UnderV.Last();	//j
		int64 Large = OverV.Last();		//i
		UnderV.DelLast();
		OverV.DelLast();
		//update tables
		KTbl[Small] = Large;	//K[j] = i
		UTbl[Large] = UTbl[Large] + UTbl[Small] - 1;	//U[i] = U[i] + U[j] - 1
		//Small (j) is now exactly full
		//assign Large (i) to under/overfull by criteria
		if (UTbl[Large] < 1)
		{
			UnderV.Add(Large);
		} 
		else
		{
			OverV.Add(Large);
		}
	}
	//set remaining entries (caused by rounding errors)
	while(UnderV.Len() > 0)
	{
		int64 curr = UnderV.Last();
		UnderV.DelLast();
		UTbl[curr]=1;
	}
	while(OverV.Len() > 0)
	{
		int64 curr = OverV.Last();
		OverV.DelLast();
		UTbl[curr]=1;
	}

	//because we passed in a pointer to the node data tables, node data for CurrI->node NI is now set to the U and K tables
}

//Get random element using alias sampling method
//NTTable is TPair<TVec<TInt, int>, TVec<TFlt, int>>, aka K and U tables
int64 AliasDrawInt(TIntVFltVPr& NTTable, TRnd& Rnd)
{
	printf(NTTable);
	int64 N = NTTable.GetVal1().Len();		//size of tables
	TInt X = static_cast<int64>(Rnd.GetUniDev()*N);	//uniform random {1, 2,...n}
	double Y = Rnd.GetUniDev();	//random  on [0,1)]
	//if Y < U[x], return x, else return K[x]
	return Y < NTTable.GetVal2()[X] ? X : NTTable.GetVal1()[X];
}

//preprocess the single node given as NI
//parameters: network, p, q, node, shared node count, display toggle flag
void PreprocessNode (PWNet& InNet, const double& ParamP, const double& ParamQ,
 TWNet::TNodeI NI, int64& NCnt, const bool& Verbose)
 {
 	//process print every hundred nodes
	if (Verbose && NCnt%100 == 0) 
		printf("\rPreprocessing progress: %.2lf%% ",(double)NCnt*100/(double)(InNet->GetNodes()));fflush(stdout);

	//for node t, build hash of neighbor id -> bool
	THash <TInt, TBool> NbrH;                                    //Neighbors of t
	for (int64 i = 0; i < NI.GetOutDeg(); i++)		//loop neighbors
	{
		NbrH.AddKey(NI.GetNbrNId(i));
	}

	for (int64 i = 0; i < NI.GetOutDeg(); i++)		//loop neighbors again
	{
		//get current neighbor to node we are processing
		TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));      //for each node v

		double Psum = 0;
		TFltV PTable;                              //Probability distribution table
		//loop neighbors of the neighbor (2-hop neighbors of processing node)
		for (int64 j = 0; j < CurrI.GetOutDeg(); j++) 
		{           //for each node x
			int64 FId = CurrI.GetNbrNId(j);		//id of neighbor-neighbor
			TFlt Weight;
			//get weight of edge between current neighbor and neighbor-neighbor
			//if edge doesn't exist, skip to next neighbor (but this should never happen?)
			if (!(InNet->GetEDat(CurrI.GetId(), FId, Weight))){ continue; }

			//compute walk bias alpha for this edge
			//append alpha to end of PTable vector and add to Psum

			//if neighbor-neighbor is processing node (ie, traveled back along same edge), alpha = weight / p
			if (FId==NI.GetId())
			{
				PTable.Add(Weight / ParamP);
				Psum += Weight / ParamP;
			} 
			//if neighbor-neighbor is connected to processing node, alpha = weight
			else if (NbrH.IsKey(FId))
			{
				PTable.Add(Weight);
				Psum += Weight;
			} 
			//otherwise (not processing node or direct neighbor of processing node), alpha = weight / q
			else
			{
				PTable.Add(Weight / ParamQ);
				Psum += Weight / ParamQ;
			}
		}
		//Normalizing table - divide all alpha values by the sum
		for (int64 j = 0; j < CurrI.GetOutDeg(); j++)
		{
			PTable[j] /= Psum;
		}

		//pass in normalized probability table, and CurrI's allocated data space for node NI
		//second argument is TPair<TVec<TInt, int>, TVec<TFlt, int>>
		GetNodeAlias(PTable, CurrI.GetDat().GetDat(NI.GetId()));		//generate U and K tables for NI stored in CurrI data
	}
	NCnt++;		//increment shared node counter
}

//Preprocess transition probabilities for each path t->v->x
//InNet - network
//ParamP, ParamQ - return and inout parameters
//Verbose - display toggle flag
void PreprocessTransitionProbs(PWNet& InNet, const double& ParamP, const double& ParamQ, const bool& Verbose)
{
	if (Verbose)
		printf("Setting node data type...\n");
	//set node data for each node to hash of int -> TIntVFltVPr, where TIntVFltVPr is TPair<TVec<TInt, int>, TVec<TFlt, int> >
	//essentially, map node id to both an integer and floating point vector, where each vector can contain max of 2 billion (2^31) elements
	for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) 	//loop nodes
	{
		InNet->SetNDat(NI.GetId(),TIntIntVFltVPrH());	
	}

	if (Verbose)
		printf("Allocating space for node data...\n");
	//allocating space in advance to avoid issues with multithreading
	for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++)		//loop nodes
	{
		for (int64 i = 0; i < NI.GetOutDeg(); i++) 		//loop neighbor indexes
		{                    
			TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));	//get neighbor
			CurrI.GetDat().AddDat(NI.GetId(),TPair<TIntV,TFltV>(TIntV(CurrI.GetOutDeg()),TFltV(CurrI.GetOutDeg())));
			//to the current neighbor node's data hash, add an element
			//node id -> int vector (length = number of neighbor's neighbors), float vector (same length)
			//ie, NI=A is connected to CurrI=B; add to B's data A -> <int vec, float vec>, with both of length outdegree(B)
			//(no wonder this is a space hog)
		}
	}
	
	//build a vector of all node ids
	TIntV NIds;
	for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++)
	{
		NIds.Add(NI.GetId());
	}
	
	//preprocess each node in the graph, in parallel
	if (Verbose)
		printf("Preprocessing nodes in parallel...\n");
	int64 NCnt = 0;
#pragma omp parallel for schedule(dynamic)
	for (int64 i = 0; i < NIds.Len(); i++)
	{
		PreprocessNode(InNet, ParamP, ParamQ, InNet->GetNI(NIds[i]), NCnt, Verbose);
		//pass in: network, P and Q params, node to process, shared counter for periodic prints, and display toggle flag
	}
	if(Verbose){ printf("\n"); }

	//all U and K tables now complete
}

int64 PredictMemoryRequirements(PWNet& InNet)
{
	int64 MemNeeded = 0;
	for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++)
	{
		for (int64 i = 0; i < NI.GetOutDeg(); i++) {
			TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
			MemNeeded += CurrI.GetOutDeg()*(sizeof(TInt) + sizeof(TFlt));
		}
	}
	return MemNeeded;
}

//Simulates a random walk, starting at node StartNId, of length WalkLen
//InNet is the graph/network
//Rnd is the previously-seeded randomizer
//WalkV will hold just this walk
void SimulateWalk(PWNet& InNet, int64 StartNId, const int& WalkLen, TRnd& Rnd, TIntV& WalkV)
{
	WalkV.Add(StartNId);		//add starting node id to random walk
	
	//if desired length is 1, done
	if (WalkLen == 1)
		return;

	//if this node has no outgoing edges, done
	if (InNet->GetNI(StartNId).GetOutDeg() == 0)
		return;

	//add random neighbor to walk
	WalkV.Add(InNet->GetNI(StartNId).GetNbrNId(Rnd.GetUniDevInt(InNet->GetNI(StartNId).GetOutDeg())));

	//keep adding until reach desired walk length
	while (WalkV.Len() < WalkLen)
	{
		int64 Dst = WalkV.Last();		//last node added to walk (current location)
		int64 Src = WalkV.LastLast();	//previous node in walk

		//if last node in path has no outgoing edges, done
		if (InNet->GetNI(Dst).GetOutDeg() == 0)
			return;

		//add another random node to walk - based on transition probabilities and alias sampling
		int64 Next = AliasDrawInt(InNet->GetNDat(Dst).GetDat(Src), Rnd);
		WalkV.Add(InNet->GetNI(Dst).GetNbrNId(Next));
	}
}
