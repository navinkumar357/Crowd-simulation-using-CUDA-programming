//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_waypoint.h"

#include <cmath>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

// initialize static variables
int Ped::Twaypoint::staticid = 0;

// Constructor: Sets some intial values. The agent has to pass within the given radius.
Ped::Twaypoint::Twaypoint(double px, double py, double pr) : id(staticid++), x(px), y(py), r(pr) {};

// Constructor - sets the most basic parameters.
Ped::Twaypoint::Twaypoint() : id(staticid++), x(0), y(0), r(1) {};

Ped::Twaypoint::~Twaypoint() {};


