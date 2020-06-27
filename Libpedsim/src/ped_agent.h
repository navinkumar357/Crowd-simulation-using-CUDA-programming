//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; }
		int getDesiredY() const { return desiredPositionY; }

		// Sets the agent's position
		void setX(int newX) { x = newX; }
		void setY(int newY) { y = newY; }

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		// Returns the next destination to visit
		Twaypoint* getNextDestination();

		// Position of agent defined by x and y
		int getX() const { return x; };
		int getY() const { return y; };

		// Region of agent defined by x and y
		int getRegionId() const { return regionId; };
		void setRegionId(int newRegionId) { regionId = newRegionId; };

		long getId() const { return id; };
		void setId(long newId) { id = newId; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		// The current destination (may require several steps to reach)
		Twaypoint* destination;

	private:
		Tagent() {};

		// The agent's current position
		int x;
		int y;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The agent's region
		int regionId;

		long id;

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void init(int posX, int posY);


	};

	class TagentSIMD {
	public:
		TagentSIMD(int nAgents) {
			x = (int *)_mm_malloc(nAgents * sizeof(int), 16);
			y = (int *)_mm_malloc(nAgents * sizeof(int), 16);
			desiredX = (int *)_mm_malloc(nAgents * sizeof(int), 16);
			desiredY = (int *)_mm_malloc(nAgents * sizeof(int), 16);
			destinationX = (float *)_mm_malloc(nAgents * sizeof(float), 16);
			destinationY = (float *)_mm_malloc(nAgents * sizeof(float), 16);

		};
		//private:
		int *x;
		int *y;
		int *desiredX;
		int *desiredY;

		float *destinationX;
		float *destinationY;

		//private:
		TagentSIMD() {}
	};
}

#endif