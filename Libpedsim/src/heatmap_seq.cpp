// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"
#include "cuda_testkernel.h"

#include <cstdlib>
#include <iostream>
using namespace std;

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

// Sets up the heatmap
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE * sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE * sizeof(int));

	heatmap = (int**)malloc(SIZE * sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
	scaled_heatmap1 = (int**)malloc(SCALED_SIZE * sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));

	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE * i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE * i;
		scaled_heatmap1[i] = shm + SCALED_SIZE * i;
		blurred_heatmap[i] = bhm + SCALED_SIZE * i;
	}

	cuda_setupHeatmap(*heatmap, *scaled_heatmap, *blurred_heatmap);
}

void Ped::Model::updateHeatmapCUDA(Model *model)
{
	int *desiredX = (int*)malloc(agents.size() * sizeof(int));
	int *desiredY = (int*)malloc(agents.size() * sizeof(int));
	for (int i = 0; i < agents.size(); i++) {
		desiredX[i] = agents[i]->getDesiredX();
		desiredY[i] = agents[i]->getDesiredY();
	}

	cuda_updateHeatmap(model, *heatmap, *scaled_heatmap, *blurred_heatmap, 1024, 5, desiredX, desiredY, agents.size());
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			// heat fades
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y][x] += 40;

	}

	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}

	// Scale the data for visual representation
	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}