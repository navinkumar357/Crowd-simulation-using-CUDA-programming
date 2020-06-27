//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <thread>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <smmintrin.h>


// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif


/*// Print the content of a __m128-variable.
void print128i_num(__m128i var)
{
uint32_t *val = (uint32_t*)&var;
printf("Numerical: %i %i %i %i \n",
val[0], val[1], val[2], val[3]);
}

void print128_num(__m128 var)
{
float *val = (float*)&var;
printf("Numerical: %f %f %f %f \n",
val[0], val[1], val[2], val[3]);
}*/


void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	agentsSIMD = Ped::TagentSIMD(static_cast<int>(agents.size()));
	for (int i = 0; i < agents.size(); i++) {
		agentsSIMD.x[i] = agents[i]->getX();
		agentsSIMD.y[i] = agents[i]->getY();
	}

	x = std::vector<__m128i>();
	y = std::vector<__m128i>();
	desiredX = std::vector<__m128i>();
	desiredY = std::vector<__m128i>();
	destinationX = std::vector<__m128>();
	destinationY = std::vector<__m128>();

	for (int i = 0; i < agents.size(); i += 4) {
		x.push_back(_mm_load_si128((__m128i *) &agentsSIMD.x[i]));
		y.push_back(_mm_load_si128((__m128i *) &agentsSIMD.y[i]));
		desiredX.push_back(_mm_load_si128((__m128i *) &agentsSIMD.desiredX[i]));
		desiredY.push_back(_mm_load_si128((__m128i *) &agentsSIMD.desiredY[i]));
		destinationX.push_back(_mm_load_ps(&agentsSIMD.destinationX[i]));
		destinationY.push_back(_mm_load_ps(&agentsSIMD.destinationY[i]));
	}


	// Assign region for all the agents and get the list of agents in each region
	// TODO: Dynamically get region boundaries
	for (int i = 0; i < agents.size(); i++) {
		if (agents[i]->getX() < 100) {
			if (agents[i]->getY() < 60) {
				region1.insert(agents[i]);
				agents[i]->setRegionId(1);
			}
			else {
				region3.insert(agents[i]);
				agents[i]->setRegionId(3);
			}
		}
		else {
			if (agents[i]->getY() < 60) {
				region2.insert(agents[i]);
				agents[i]->setRegionId(2);
			}
			else {
				region4.insert(agents[i]);
				agents[i]->setRegionId(4);
			}
		}
	}

	for (int i = 0; i < agents.size(); i++) {
		agents[i]->setId(i);
	}


	// TODO: change 300 to something else? Remember to change it in the tick-function as well.
	vector<long> v(300, -1);
	coordinates = vector<vector<long>>(300, v);

	for (int i = 0; i < agents.size(); i++) {
		coordinates[agents[i]->getX()][agents[i]->getY()] = agents[i]->getId();
	}



	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void thread_imp(int thread_id, std::vector<Ped::Tagent*> agents, int num_threads) {

	for (int i = thread_id * agents.size() / num_threads; i < (thread_id + 1) * agents.size() / num_threads; ++i) {
		agents[i]->computeNextDesiredPosition();
		int newX = agents[i]->getDesiredX();
		int newY = agents[i]->getDesiredY();
		agents[i]->setX(newX);
		agents[i]->setY(newY);
	}
}


void Ped::Model::tick_SIMD() {
	// Compute the destination for all agents and store it in the destination array for SIMD
	for (int i = 0; i < agents.size(); i++) {
		agents[i]->destination = agents[i]->getNextDestination();
		if (agents[i]->destination == NULL) {
			// no destination, no need to compute where to move to
			continue;
		}
		agentsSIMD.destinationX[i] = agents[i]->destination->getx();
		agentsSIMD.destinationY[i] = agents[i]->destination->gety();
	}

	// Load the destination array into SIMD vector of size 4 at a time
	int vector_index = 0;
	for (int i = 0; i < agents.size(); i += 4) {
		destinationX[vector_index] = _mm_load_ps(&agentsSIMD.destinationX[i]);
		destinationY[vector_index] = _mm_load_ps(&agentsSIMD.destinationY[i]);
		vector_index++;
	}

	// Compute next desired position using SIMD vectorisation
	for (int i = 0; i < x.size(); i++) {
		__m128 x_double = _mm_cvtepi32_ps(x[i]);
		__m128 y_double = _mm_cvtepi32_ps(y[i]);

		__m128 diffX = _mm_sub_ps(destinationX[i], x_double);
		__m128 diffY = _mm_sub_ps(destinationY[i], y_double);

		__m128 diffXSquared = _mm_mul_ps(diffX, diffX);
		__m128 diffYSquared = _mm_mul_ps(diffY, diffY);

		__m128 lengthSquared = _mm_add_ps(diffXSquared, diffYSquared);
		__m128 length = _mm_sqrt_ps(lengthSquared);

		__m128 divX = _mm_div_ps(diffX, length);
		__m128 divY = _mm_div_ps(diffY, length);

		__m128 dPositionX = _mm_add_ps(x_double, divX);
		__m128 dPositionY = _mm_add_ps(y_double, divY);

		__m128i desiredPositionX = _mm_cvtps_epi32(dPositionX);
		__m128i desiredPositionY = _mm_cvtps_epi32(dPositionY);

		// Store the results in the vectors.
		_mm_store_si128(&desiredX[i], desiredPositionX);
		_mm_store_si128(&desiredY[i], desiredPositionY);

		_mm_store_si128(&x[i], desiredPositionX);
		_mm_store_si128(&y[i], desiredPositionY);
	}

	vector_index = 0;
	for (int i = 0; i < agents.size(); i += 4) {
		_mm_store_si128((__m128i *) &agentsSIMD.desiredX[i], desiredX[vector_index]);
		_mm_store_si128((__m128i *) &agentsSIMD.desiredY[i], desiredY[vector_index]);
		vector_index++;
	}

	// Update the new coordinates for the agents and agents SIMD.
	for (int i = 0; i < agents.size(); ++i) {
		agentsSIMD.x[i] = agentsSIMD.desiredX[i];
		agentsSIMD.y[i] = agentsSIMD.desiredY[i];
		agents[i]->setX(agentsSIMD.desiredX[i]);
		agents[i]->setY(agentsSIMD.desiredY[i]);
	}
}

void Ped::Model::tick_SIMDOMP() {
	omp_set_num_threads(4);
	// Compute the destination for all agents and store it in the destination array for SIMD
#pragma omp parallel for
	for (int i = 0; i < agents.size(); i++) {
		agents[i]->destination = agents[i]->getNextDestination();
		if (agents[i]->destination == NULL) {
			// no destination, no need to compute where to move to
			continue;
		}
		agentsSIMD.destinationX[i] = agents[i]->destination->getx();
		agentsSIMD.destinationY[i] = agents[i]->destination->gety();
	}

	// Load the destination array into SIMD vector of size 4 at a time
	int vector_index = 0;
	for (int i = 0; i < agents.size(); i += 4) {
		destinationX[vector_index] = _mm_load_ps(&agentsSIMD.destinationX[i]);
		destinationY[vector_index] = _mm_load_ps(&agentsSIMD.destinationY[i]);
		vector_index++;
	}

	// Compute next desired position using SIMD vectorisation
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {
		__m128 x_double = _mm_cvtepi32_ps(x[i]);
		__m128 y_double = _mm_cvtepi32_ps(y[i]);

		__m128 diffX = _mm_sub_ps(destinationX[i], x_double);
		__m128 diffY = _mm_sub_ps(destinationY[i], y_double);

		__m128 diffXSquared = _mm_mul_ps(diffX, diffX);
		__m128 diffYSquared = _mm_mul_ps(diffY, diffY);

		__m128 lengthSquared = _mm_add_ps(diffXSquared, diffYSquared);
		__m128 length = _mm_sqrt_ps(lengthSquared);

		__m128 divX = _mm_div_ps(diffX, length);
		__m128 divY = _mm_div_ps(diffY, length);

		__m128 dPositionX = _mm_add_ps(x_double, divX);
		__m128 dPositionY = _mm_add_ps(y_double, divY);

		__m128i desiredPositionX = _mm_cvtps_epi32(dPositionX);
		__m128i desiredPositionY = _mm_cvtps_epi32(dPositionY);

		// Store the results in the vectors.
		_mm_store_si128(&desiredX[i], desiredPositionX);
		_mm_store_si128(&desiredY[i], desiredPositionY);

		_mm_store_si128(&x[i], desiredPositionX);
		_mm_store_si128(&y[i], desiredPositionY);
	}

	vector_index = 0;
	for (int i = 0; i < agents.size(); i += 4) {
		_mm_store_si128((__m128i *) &agentsSIMD.desiredX[i], desiredX[vector_index]);
		_mm_store_si128((__m128i *) &agentsSIMD.desiredY[i], desiredY[vector_index]);
		vector_index++;
	}

	//cout << &agentsSIMD. << endl;
	// Update the new coordinates for the agents and agents SIMD.
#pragma omp parallel for
	for (int i = 0; i < agents.size(); ++i) {
		agentsSIMD.x[i] = agentsSIMD.desiredX[i];
		agentsSIMD.y[i] = agentsSIMD.desiredY[i];
		agents[i]->setX(agentsSIMD.desiredX[i]);
		agents[i]->setY(agentsSIMD.desiredY[i]);
	}
}

class Region {
public:
	int lowerX;
	int upperX;
	int lowerY;
	int upperY;
	Region(int lx, int ux, int ly, int uy) {
		lowerX = lx;
		upperX = ux;
		lowerY = ly;
		upperY = uy;
	}
};

void Ped::Model::collision_detection_regions() {
	omp_set_num_threads(4);

#pragma omp parallel
	{
#pragma omp sections nowait
		{
#pragma omp section
			{
				regionTask(region1);
			}
#pragma omp section
			{
				regionTask(region2);
			}
#pragma omp section
			{
				regionTask(region3);
			}
#pragma omp section
			{
				regionTask(region4);
			}
		}
	}


	// TODO: do something more efficient than this?
	region1.clear();
	region2.clear();
	region3.clear();
	region4.clear();

	// Update the agents new region based on new X and Y values and region sets
	for (int i = 0; i < agents.size(); i++) {
		if (agents[i]->getX() < 100) {
			if (agents[i]->getY() < 60) {
				region1.insert(agents[i]);
				agents[i]->setRegionId(1);
			}
			else {
				region3.insert(agents[i]);
				agents[i]->setRegionId(3);
			}
		}
		else {
			if (agents[i]->getY() < 60) {
				region2.insert(agents[i]);
				agents[i]->setRegionId(2);
			}
			else {
				region4.insert(agents[i]);
				agents[i]->setRegionId(4);
			}

		}
	}
}


void Ped::Model::regionTask(set<Tagent*> region) {
	for (set<Tagent *>::iterator it = region.begin(); it != region.end(); ++it) {
		(*it)->computeNextDesiredPosition();
		moveRegions(*it);
	}
}

void Ped::Model::tick()
{
	if (this->implementation == SEQ) {
		//Serial Code
		for (int i = 0; i < agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			int newX = agents[i]->getDesiredX();
			int newY = agents[i]->getDesiredY();
			agents[i]->setX(newX);
			agents[i]->setY(newY);
			//move(agents[i]);
		}

		/*vector<Region> regions = vector<Region>();
		regions.push_back(Region(0, 99, 0, 59));
		regions.push_back(Region(100, 199, 0, 59));
		regions.push_back(Region(0, 99, 60, 119));
		regions.push_back(Region(100, 199, 60, 119));*/

	}
	else if (this->implementation == OMP) {
		// OpenMP Code
		omp_set_num_threads(4);
#pragma omp parallel for
		for (int i = 0; i < agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			int newX = agents[i]->getDesiredX();
			int newY = agents[i]->getDesiredY();
			agents[i]->setX(newX);
			agents[i]->setY(newY);
		}
	}
	else if (this->implementation == VECTOR) {
		// SIMD
		tick_SIMD();
	}
	else if (this->implementation == VECTOROMP) {
		// SIMD + OpenMP
		tick_SIMDOMP();
	}
	else if (this->implementation == PTHREAD) {
		// Pthread C++ Code
		const int NUM_THREADS = 1;
		thread threads[NUM_THREADS];
		for (int i = 0; i < NUM_THREADS; i++) {
			threads[i] = thread(thread_imp, i, agents, NUM_THREADS);
		}

		for (int i = 0; i < NUM_THREADS; i++) {
			threads[i].join();
		}
	}
	else if (this->implementation == CUDA) {
		// CUDA
		for (int i = 0; i < agents.size(); i++) {
			agents[i]->destination = agents[i]->getNextDestination();
			if (agents[i]->destination == NULL) {
				// no destination, no need to compute where to move to
				continue;
			}
			agentsSIMD.destinationX[i] = agents[i]->destination->getx();
			agentsSIMD.destinationY[i] = agents[i]->destination->gety();
		}

		Tuple ret = cuda_tick(agentsSIMD.x, agentsSIMD.y,
			agentsSIMD.destinationX, agentsSIMD.destinationY,
			agentsSIMD.desiredX, agentsSIMD.desiredY, agents.size());

		for (int i = 0; i < agents.size(); i++) {
			agentsSIMD.desiredX[i] = ret.desiredX[i];
			agentsSIMD.desiredY[i] = ret.desiredY[i];
			agentsSIMD.x[i] = ret.desiredX[i];
			agentsSIMD.y[i] = ret.desiredY[i];
			agents[i]->setX(ret.desiredX[i]);
			agents[i]->setY(ret.desiredY[i]);
		}
	}
	else if (this->implementation == SEQCOLLISION) {
		for (int i = 0; i < agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			move(agents[i]);
		}
	}
	else if (this->implementation == SEQCOLLISIONOMP) {
		omp_set_num_threads(4);
#pragma omp parallel for
		for (int i = 0; i < agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			move(agents[i]);
		}
	}
	else if (this->implementation == REGION) {
		collision_detection_regions();
	}
	else if (this->implementation == HEATMAP_SEQ) {
		collision_detection_regions();
		auto start = std::chrono::steady_clock::now();
		updateHeatmapSeq();
		auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
		cout << "Target time: " << duration_target.count() << " milliseconds, " << std::endl;
	}
	else if (this->implementation == CPU_GPU) {
		updateHeatmapCUDA(this);
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////


// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}


// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::moveRegions(Ped::Tagent *agent)
{
	// Search for neighboring agents
	//set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);
	set<const Ped::Tagent *> neighbors = getNeighborsRegions(agent, 4);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;

	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			long newAgentId = agent->getId();
			long oldAgentId = coordinates[(*it).first][(*it).second];

			// If there is an agent here: try the next move (if there is one).
			if (oldAgentId != -1) {
				continue;
			}

			// If the new coordinates are free, mark them as occupied using CAS.
			if (_InterlockedCompareExchange(&coordinates[(*it).first][(*it).second], newAgentId, oldAgentId) == oldAgentId) {

				// Mark the agent's previous coordinates as free.
				if (_InterlockedCompareExchange(&coordinates[agent->getX()][agent->getY()], -1, agent->getId()) != agent->getId()) {
					cout << "CAS for updating free coordinate failed" << endl;
				}

				// Update the agent's position.
				agent->setX((*it).first);
				agent->setY((*it).second);
				return;
			}
			else {
				cout << "CAS for updating agent movement coordinate failed" << endl;
			}
		}
	}
}

// Return the distance between two points in a 2d-plane.
double getDistance(int x1, int y1, int x2, int y2) {
	//return sqrt(((x2 - x1)*(x2 - x1)) + ((y2 - y1)*(y2 - y1)));
	int diffX = (x1 >= x2) ? x1 - x2 : x2 - x1;
	int diffY = (y1 >= y2) ? y1 - y2 : y2 - y1;
	return diffX + diffY;
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
//set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
set<const Ped::Tagent*> Ped::Model::getNeighborsRegions(Tagent *agent, int dist) const {

	// Pick the correct region
	set<Tagent *> region;
	if (agent->getRegionId() == 1) {
		region = region1;
	}
	else if (agent->getRegionId() == 2) {
		region = region2;
	}
	else if (agent->getRegionId() == 3) {
		region = region3;
	}
	else if (agent->getRegionId() == 4) {
		region = region4;
	}

	// Fetch all the neighbors within the specified distance.
	set<const Ped::Tagent *> neighbors;
	for (set<Tagent *>::iterator it = region.begin(); it != region.end(); ++it) {
		if (getDistance((*it)->getX(), (*it)->getY(), agent->getX(), agent->getY()) <= dist) {
			neighbors.insert(*it);
		}
	}

	return neighbors;
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) {delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) {delete destination; });
}