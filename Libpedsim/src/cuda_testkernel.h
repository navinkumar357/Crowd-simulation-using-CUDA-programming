#include "ped_model.h"

#pragma once
struct Tuple {
	int* desiredX;
	int* desiredY;
};

int cuda_test();
void cuda_setupHeatmap(int *heatmap, int *scaled_heatmap, int *blurred_heatmap);
void cuda_updateHeatmap(Ped::Model *model, int *heatmap, int *scaled_heatmap, int *blurred_heatmap, int size, int cell_size, int *desiredX, int *desiredY, int agents);
Tuple cuda_tick(const int *x, const int *y, const float *destinationX, const float *destinationY, int *desiredX, int *desiredY, int size);