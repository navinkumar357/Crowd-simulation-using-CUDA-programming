//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <smmintrin.h>

#include "ped_agent.h"

namespace Ped {
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION {
		CUDA, VECTOR, OMP, PTHREAD, SEQ, VECTOROMP, REGION, SEQCOLLISION, SEQCOLLISIONOMP, DYNAMICREGION, CPU_GPU, HEATMAP_SEQ
	};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation);

		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();
		void regionTask(set<Tagent*>);
		void region2Task();
		void region3Task();
		void region4Task();
		void collision_detection_regions();

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		set<const Ped::Tagent*> getNeighborsRegions(Tagent * agent, int dist) const;

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		Ped::TagentSIMD agentsSIMD;
		std::vector<__m128i> x;
		std::vector<__m128i> y;
		std::vector<__m128i> desiredX;
		std::vector<__m128i> desiredY;
		std::vector<__m128> destinationX;
		std::vector<__m128> destinationY;
		void tick_SIMD();
		void tick_SIMDOMP();

		set<Tagent*>region1 = set<Tagent*>();
		set<Tagent*>region2 = set<Tagent*>();
		set<Tagent*>region3 = set<Tagent*>();
		set<Tagent*>region4 = set<Tagent*>();



		vector<set<Tagent *>> regions = vector<set<Tagent *>>();


		vector<vector<long>> coordinates;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		void moveRegions(Ped::Tagent * agent);

		void setAgentPosition(Tagent * agent);

		// TODO: Add a datastructure to store the structure of array of agents


		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		//set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;
		set<const Ped::Tagent*> getNeighbors(Tagent *agent, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;
		int ** scaled_heatmap1;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapCUDA(Model *model);
		void updateHeatmapSeq();
	};
}
#endif