/*************************************************************************
    > File Name: main.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Tue 31 Dec 2019 05:36:12 PM CST
 ************************************************************************/

#include "lrsolver.h"

#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cout << "Usage: ./lr [train, test] --input-file [input file]"
      <<  " --num-features [feature numbers]"
      <<  " --num-samples [sample numbers]"
      <<  " --learning-rate [learning_rate]" << std::endl;
  }

  bool willTrain = false;
  if (!strcmp("train", argv[1])) {
    willTrain = true;
  }

  if (willTrain) {
    std::cout << "LR: training start." << std::endl;
    std::string inputFile;
    std::string outputFile = "out.weights";
    int numFeature;
    int numSamples;
    float learningRate = 0.01;
    int numIterations = 10;
    for (int i = 0; i < argc; ++i) {
      if (!strcmp("--input-file", argv[i])) {
        assert(i+1 < argc);
        inputFile = std::string(argv[++i]);
      } else if (!strcmp("--num-features", argv[i])) {
        assert(i+1 < argc);
        numFeature = std::stoi(argv[++i]);
      } else if (!strcmp("--num-samples", argv[i])) {
        assert(i+1 < argc);
        numSamples = std::stoi(argv[++i]);
      } else if (!strcmp("--learning-rate", argv[i])) {
        assert(i+1 < argc);
        learningRate = std::stof(argv[++i]);
      } else if (!strcmp("--num-iterations", argv[i])) {
        assert(i+1 < argc);
        numIterations = std::stoi(argv[++i]);
      } else if (!strcmp("--output-file", argv[i])) {
        assert(i+1 < argc);
        outputFile = std::string(argv[++i]);
      }
    }

    LRSolver* solver = LRSolver::fromTraingSet(inputFile, numFeature, numSamples, learningRate);
    solver->train(numIterations);
    solver->dumpWeights(outputFile);
    delete solver;
  } else {
    std::string weightsFile;
    std::string evaluateList;
    float threshold;
    int numFeature;
    for (int i = 0; i < argc; ++i) {
      if (!strcmp("--weights-file", argv[i])) {
        assert(i+1 < argc);
        weightsFile = std::string(argv[++i]);
      } else if (!strcmp("--evaluate-list", argv[i])) {
        assert(i+1 < argc);
        evaluateList = std::string(argv[++i]);
      } else if (!strcmp("--threshold", argv[i])) {
        assert(i+1 < argc);
        threshold = std::stof(argv[++i]);
      } else if (!strcmp("--num-features", argv[i])) {
        assert(i+1 < argc);
        numFeature = std::stoi(argv[++i]);
      }
    }


    int numSamples;
    std::ifstream ifs;
    ifs.open(evaluateList, std::ifstream::in);
    if (!ifs.good()) {
      std::cerr << "LRSolver: File I/O error." << std::endl;
      ifs.close();
      return 0;
    }

    ifs >> numSamples;
    float* y = new float[numSamples];
    float** x = new float*[numSamples];
    for (int i = 0; i < numSamples; ++i) {
      x[i] = new float[numFeature+1];
      x[i][0] = 1.0f;
      for (int j = 1; j < numFeature+1; ++j) {
        ifs >> x[i][j];
      }
      ifs >> y[i];
    }
    LRSolver* solver = LRSolver::fromWeights(weightsFile, numFeature, threshold);
    solver->evaluate(x, y, numSamples);

    delete [] x;
    delete [] y;
    delete solver;
  }

  return 0;
}
