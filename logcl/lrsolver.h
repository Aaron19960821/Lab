/*************************************************************************
    > File Name: lrsolver.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Tue 31 Dec 2019 05:37:00 PM CST
 ************************************************************************/

#ifndef LR_SOLVER_H_
#define LR_SOLVER_H_

#include <string>

class LRSolver {
  public:
    LRSolver();

    static LRSolver* fromTraingSet(const std::string& inputFile,
        int numFeature,
        int numTrainingSample,
        float learningRate);
    static LRSolver* fromWeights(const std::string& weightsFile,
        int numFeature, float threshHold);

    void train(int numInterations);
    void dumpWeights(const std::string& outputFile);

    void evaluate(float** x, float* y, int numSamples);
    ~LRSolver();

    LRSolver(float* x, 
        float* y, 
        int numFeature, 
        int numTrainingSample, 
        float learningRate);

    LRSolver(float* weights,
        int numFeature,
        float threshHold);

  private:
    float* _x;
    float* _y;
    int _numFeature;
    int _numTrainingSamples;
    float _learningRate;

    float _threshHold;

    float* weights;

};

#endif
