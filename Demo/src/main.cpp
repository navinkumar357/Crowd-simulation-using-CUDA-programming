///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// 
//
// The main starting point for the crowd simulation.
//



#undef max
#include <Windows.h>
#include "ped_model.h"
#include "MainWindow.h"
#include "ParseScenario.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include <thread>

#include "PedSimulation.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

int main(int argc, char*argv[]) {
	bool timing_mode = 0;
	int i = 1;
	QString scenefile = "scenario.xml";
	//QString scenefile = "scenario_box.xml";
	// QString scenefile = "hugeScenario.xml";
	int mode = 10;
	// Enable memory leak check. This is ignored when compiling in Release mode. 
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// Argument handling
	while (i < argc)
	{
		if (argv[i][0] == '-' && argv[i][1] == '-')
		{
			if (strcmp(&argv[i][2], "timing-mode") == 0)
			{
				cout << "Timing mode on\n";
				timing_mode = true;
			}
			if (strcmp(&argv[i][2], "openmp") == 0)
			{
				cout << "openMP arg parsing\n";
				mode = 2;
			}
			if (strcmp(&argv[i][2], "pthread") == 0)
			{
				mode = 3;
			}
			if (strcmp(&argv[i][2], "vector") == 0)
			{
				mode = 4;
			}
			if (strcmp(&argv[i][2], "vectoromp") == 0)
			{
				mode = 5;
			}
			if (strcmp(&argv[i][2], "cuda") == 0)
			{
				mode = 6;
			}
			if (strcmp(&argv[i][2], "nocollisionseq") == 0)
			{
				mode = 7;
			}
			if (strcmp(&argv[i][2], "nocollisionregion") == 0)
			{
				mode = 8;
			}
			if (strcmp(&argv[i][2], "nocollisionseqomp") == 0)
			{
				mode = 9;
			}
			if (strcmp(&argv[i][2], "heatmapseq") == 0)
			{
				mode = 10;
			}
			if (strcmp(&argv[i][2], "heatmapparallel") == 0)
			{
				mode = 11;
			}
			else if (strcmp(&argv[i][2], "help") == 0)
			{
				cout << "Usage: " << argv[0] << " [--help] [--timing-mode] [scenario]" << endl;
				return 0;
			}
			else
			{
				cerr << "Unrecognized command: \"" << argv[i] << "\". Ignoring ..." << endl;
			}
		}
		else // Assume it is a path to scenefile
		{
			scenefile = argv[i];
		}

		i += 1;
	}
	int retval = 0;
	{ // This scope is for the purpose of removing false memory leak positives

	  // Reading the scenario file and setting up the crowd simulation model
		Ped::Model model;
		ParseScenario parser(scenefile);
		model.setup(parser.getAgents(), parser.getWaypoints(), Ped::HEATMAP_SEQ);

		// GUI related set ups
		QApplication app(argc, argv);
		MainWindow mainwindow(model);

		// Default number of steps to simulate. Feel free to change this.
		const int maxNumberOfStepsToSimulate = 1000;



		// Timing version
		// Run twice, without the gui, to compare the runtimes.
		// Compile with timing-release to enable this automatically.
		if (timing_mode)
		{
			// Run sequentially

			double fps_seq, fps_target;
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), Ped::SEQCOLLISION);
				PedSimulation simulation(model, mainwindow);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running reference version SEQCOLLISION...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
				auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_seq = ((float)simulation.getTickCount()) / ((float)duration_seq.count())*1000.0;
				cout << "Reference time: " << duration_seq.count() << " milliseconds, " << fps_seq << " Frames Per Second." << std::endl;
			}
			Ped::IMPLEMENTATION implementation_to_test;
			switch (mode)
			{
			case 2:
				// Change this variable when testing different versions of your code. 
				// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::OMP;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version OPENMP...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs OpenMP: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 3: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::PTHREAD;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version PThread...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs Pthread: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 4: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::VECTOR;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version VECTOR...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs VECTOR: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 5: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::VECTOROMP;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version VECTOROMP...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs VECTOROMP: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 6: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::CUDA;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version CUDA...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs CUDA: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 7: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::SEQCOLLISION;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version SEQCOLLISION...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs SEQCOLLISION: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 8: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::REGION;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version REGION...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs REGION: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 9: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::SEQCOLLISIONOMP;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version SEQCOLLISIONOMP...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs SEQCOLLISIONOMP: " << fps_target / fps_seq << std::endl;
				}
				break;
			case 10: // code to be executed if n = 2;
					// Change this variable when testing different versions of your code. 
					// May need modification or extension in later assignments depending on your implementations
				implementation_to_test = Ped::HEATMAP_SEQ;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version SEQCOLLISIONOMP...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs HEATMApSEQ: " << fps_target / fps_seq << std::endl;
				}
				break;
			default: // code to be executed if n doesn't match any cases
				implementation_to_test = Ped::SEQ;
				{
					Ped::Model model;
					ParseScenario parser(scenefile);
					model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test);
					PedSimulation simulation(model, mainwindow);
					// Simulation mode to use when profiling (without any GUI)
					std::cout << "Running target version SEQ...\n";
					auto start = std::chrono::steady_clock::now();
					simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
					auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
					fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
					cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
					std::cout << "\n\nSpeedup for Seq Vs Seq: " << fps_target / fps_seq << std::endl;
				}
			}






		}
		// Graphics version
		else
		{

			PedSimulation simulation(model, mainwindow);

			cout << "Demo setup complete, running ..." << endl;

			// Simulation mode to use when visualizing
			auto start = std::chrono::steady_clock::now();
			mainwindow.show();
			simulation.runSimulationWithQt(maxNumberOfStepsToSimulate);
			retval = app.exec();

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
			float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
			cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;

		}




	}
	_CrtDumpMemoryLeaks();

	cout << "Done" << endl;
	cout << "Type Enter to quit.." << endl;
	getchar(); // Wait for any key. Windows convenience...
	return retval;
}