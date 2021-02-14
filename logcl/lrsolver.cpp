/*************************************************************************
    > File Name: lrsolver.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Thu 02 Jan 2020 05:12:00 PM CST
 ************************************************************************/

#include "lrsolver.h"

#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

LRSolver* LRSolver::fromTraingSet(const std::string& inputFile,
    int numFeature,
    int numTrainingSample,
    float learningRate) {

  std::cout << "LRSolver: Build from training set: " << inputFile << std::endl;

  std::ifstream ifs;
  ifs.open(inputFile, std::ifstream::in);
  if (!ifs.good()) {
    std::cerr << "LRSolver: File I/O error " << inputFile <<std::endl;
    ifs.close();
    return nullptr;
  }

  float* y = new float[numTrainingSample];
  float* x = new float[numTrainingSample * (numFeature + 1)];
  for (int i = 0; i < numTrainingSample; ++i) {
    x[i*(numFeature+1)] = 1.0;
    for (int j = 0; j < numFeature; ++j) {
      ifs >> x[i*(numFeature+1) + j+1];
    }
    ifs >> y[i];
  }
  ifs.close();

  return new LRSolver(x, y, numFeature, numTrainingSample, learningRate);
}

LRSolver* LRSolver::fromWeights(const std::string &weightsFile, 
    int numFeature,
    float threshHold) {
  std::ifstream ifs;
  ifs.open(weightsFile, std::ifstream::in);
  if (!ifs.good()) {
    std::cerr << "LRSolver: File I/O error." << std::endl;
    ifs.close();
    return nullptr;
  }
  float* weights = new float[numFeature+1];
  for(int i = 0; i < numFeature+1; ++i)
    ifs >> weights[i];

  return new LRSolver(weights, numFeature, threshHold);
}

LRSolver::LRSolver(float* x,
    float* y,
    int numFeature,
    int numTrainingSample,
    float learningRate): _x(x),
  _y(y),
  _numFeature(numFeature),
  _numTrainingSamples(numTrainingSample),
  _learningRate(learningRate),
  _threshHold(0.5) {
    weights = new float[numFeature + 1];
  }

LRSolver::LRSolver(float* _weights, int numFeature, float threshHold):weights(_weights),
  _numFeature(numFeature),
  _threshHold(threshHold) {
    _x = nullptr;
    _y = nullptr;
  }

LRSolver::~LRSolver() {
  delete [] _y;
  delete [] _x;
  delete [] weights;
}

void LRSolver::train(int numInterations) {

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  weights[0] = 1;
  for (int i = 1; i < _numFeature+1; ++i) {
    weights[i] = distribution(generator);
  }

  float* tmpX = new float[_numFeature+1];
  float* tmpG = new float[_numFeature+1];
  float tmpWDotX;
  for (int iter = 0; iter < numInterations; ++iter) {
    memset(tmpG, 0, (_numFeature+1)*sizeof(float));
    for (int i = 0; i < _numTrainingSamples; ++i) {
      memcpy(tmpX, &_x[i*(_numFeature+1)], (_numFeature+1)*sizeof(float));
      tmpWDotX = 0.0f;
      for (int j = 0; j < _numFeature; ++j) {
        tmpWDotX += _y[i] * weights[j] * tmpX[j];
        tmpX[j] *= _y[i];
      }
      for (int j = 0; j < _numFeature; ++j) {
        tmpG[j] += tmpX[j] / (1+exp(tmpWDotX));
      }
    }

    float loss = 0.0f;
    for (int j = 0; j < _numFeature; ++j) {
      weights[j] += _learningRate * tmpG[j] / _numTrainingSamples;
      loss += tmpG[j] * tmpG[j];
    }

    std::cout << "Iter #" << iter << ": " << sqrt(loss) << std::endl;
  }

  delete[] tmpX;
  delete[] tmpG;

  return;
}

void LRSolver::dumpWeights(const std::string& outputFile) {
  std::ofstream ofs;
  ofs.open(outputFile, std::ofstream::out | std::ofstream::app);
  std::cout << _numFeature << std::endl;
  for (int i = 0; i < _numFeature+1; ++i)
    ofs << weights[i] << std::endl;
  return;
}

void LRSolver::evaluate(float** x, float *y, int numSamples) {
  float error = 0.0;
  float tmpY;
  for (int i = 0; i < numSamples; ++i) {
    tmpY = 0.0;
    for (int j = 0; j < _numFeature+1; ++j) {
      tmpY += weights[j] * x[i][j];
    }
    if (tmpY > _threshHold) {
      tmpY = 1.0;
    } else {
      tmpY = -1.0;
    }
    
    error += fabs(tmpY-y[i]) / 2.0f;
  }

  std::cout << "Error: " << error / numSamples << std::endl;
}
