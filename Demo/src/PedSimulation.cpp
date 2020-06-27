///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== There is no need to change this file ====
// 

#include "PedSimulation.h"
#include <iostream>
#include <QApplication>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

using namespace std;

PedSimulation::PedSimulation(Ped::Model &model_, MainWindow &window_) : model(model_), window(window_), maxSimulationSteps(-1)
{
	tickCounter = 0;
}

int PedSimulation::getTickCount() const
{
	return tickCounter;
}
void PedSimulation::simulateOneStep()
{
	tickCounter++;
	model.tick();
	window.paint();
	if (maxSimulationSteps-- == 0)
	{
		QApplication::quit();
	}
}

void PedSimulation::runSimulationWithQt(int maxNumberOfStepsToSimulate)
{
	maxSimulationSteps = maxNumberOfStepsToSimulate;

	//movetimer.setInterval(50); // Limits the simulation to 20 FPS (if one so whiches).
	QObject::connect(&movetimer, SIGNAL(timeout()), this, SLOT(simulateOneStep()));
	movetimer.start();
}

void PedSimulation::runSimulationWithoutQt(int maxNumberOfStepsToSimulate)
{
	maxSimulationSteps = maxNumberOfStepsToSimulate;
	for (int i = 0; i < maxSimulationSteps; i++)
	{
		tickCounter++;
		model.tick();
	}
}
