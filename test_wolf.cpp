// test_wolf.cpp -- main program to test LH Trees using WOLF approximation for machine learning
/*
Copyright (c) 2024 ***
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/



#include <iostream>
#include <fstream>
#include <chrono>
#include <assert.h>
#include "wolf.h"
#include <vector>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <random>
#include <unordered_map>
const char* banknote_data_path = "..\\Data\\data_banknote_authentication.txt";  // banknote dataset file path

// Used to store training set features and labels
std::vector<std::vector<double>> train_features;
std::vector<uint8_t> train_labels;

// Used to store test set features and labels
std::vector<std::vector<double>> test_features;
std::vector<uint8_t> test_labels;


// Create a new std::vector<uint8_t>, taking the first 100 elements, for testing inference speed
std::vector<uint8_t> first_100_elements;
std::vector<std::vector<double>> first_100_elements_train;


// uint8_t* train_labels;
size_t treeNo; // Index of an LF_tree
extern long leafCount;
extern bool treeFinal; // an SB does not change after it is made final, treeFinal means all SBs in the tree are final
class cLF_tree;
uint32_t numberofimages = 1097;
long nSensors;
int numberLF_trees;
int minSamples2Split;
int constancyLimit;
void loadbanknoteData();
double reinforcement(uint8_t action, uint8_t label);
void testOnTRAINING_SET(int numberLF_trees, cLF_tree** apLF_tree);
void testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree);
void testTime(int numberLF_trees, cLF_tree** apLF_tree);

int main(int argc, char* argv[])
{
    std::cout << "LH Trees using WOLF approximation for machine learning\n\nInput parameters:\n" << std::endl;
    std::cout << "\nProject name " << argv[1] << "\nNumber of features                                                         " << argv[2] <<
        "\nNumber of LH Trees (multiple of 2 for the banknote dateset)                " << argv[3] << "\nMinimum images(samples) per block allowing split                           " <<
        argv[4] << "\nSensor constant if std. dev. less than                                    " << argv[5] <<
 std::endl;
    int argCount = argc;
    if (argc != 6) // We expect arguments as listed here (argc is not counted among them)
    {
        std::cout << " Wrong number of input values! " << std::endl;
    }
    char* ProjectName = argv[1];
    nSensors = atoi(argv[2]);
    numberLF_trees = atoi(argv[3]); // We grow a number of trees, some for different tasks, some to increase accuracy for a task.
    minSamples2Split = atoi(argv[4]); // A block with fewer samples can't split
    constancyLimit = atoi(argv[5]); // The closer this is to zero, the fewer variables are removed as "deemed constant"
    std::cout << "banknote dataset classification\n" << std::endl;
    loadbanknoteData();
    // Creating LF_trees  ***************************************************************************
    initCentroidSampleNumbers(nSensors); //sets up two vectors: initC and image_numbers_setup that initialize trees
    // Using several trees improves the statistical average when the trees are shifted copies of the original tree or at least different
    std::cout << "Creating array of " << numberLF_trees << " LH Trees.\n";
    auto apLF_tree = (cLF_tree**)malloc(numberLF_trees * sizeof(cLF_tree*));
    auto start_first_tree = std::chrono::steady_clock::now();
    double tparallel = 0; // This is the maximum time to grow a single LH Tree during program execution.
    // In a parallel system, this would be the time growing all trees. For example if growing 200 LF-trees takes 40 minutes
    // to learn to classify the 60000 MNIST numerals, growing the trees in parallel would take only 12seconds, and much less if the parallel code were optimized.
    for (treeNo = 0; treeNo < numberLF_trees; ++treeNo)
    {
        if (treeNo < numberLF_trees)
        {
            apLF_tree[treeNo] = create_LF_tree();
            apLF_tree[treeNo]->setTreeNumber(treeNo);
            apLF_tree[treeNo]->setSensorNumber(nSensors);
            int SBcount = 0; // SB stands for Sample Block. It is a block in the partition created by the hyperplanes of a LH Tree.
            // Grow an LF_tree  **************************************
           // std::cout << "\nStarting growth of LH Tree number " << treeNo << std::endl;
            const auto start_tree = std::chrono::steady_clock::now();
            treeFinal = false;
            //apLF_tree[treeNo]->loadSBs(1097); // loads all 60000 image numbers into SB number 0 of the new tree
            apLF_tree[treeNo]->growTree();
            //apLF_tree[treeNo]->checkFinalTree(); // Optional -- this provides a rough idea of perfomance on the training data.
            const auto finish_tree = std::chrono::steady_clock::now();
            const auto elapsed0 = std::chrono::duration<double, std::milli>(finish_tree - start_tree);
            if (elapsed0.count() > tparallel) tparallel = (double) elapsed0.count();
            auto elapsed1 = std::chrono::duration_cast<std::chrono::seconds>(finish_tree - start_first_tree);
            std::cout << "Growing tree " << treeNo << " took "  << elapsed0.count() << " millisec. Elapsed " << elapsed1.count() <<
                " sec. To go (est.) " << ceil(elapsed1.count() * (numberLF_trees - treeNo - 1)/((treeNo +1) * 60.0)) << " min." << std::endl;
        }
      } // End of for loop growing a number of LF_trees
    auto end_last_tree = std::chrono::steady_clock::now();
    auto trainTime = std::chrono::duration_cast<std::chrono::seconds>(end_last_tree - start_first_tree);
    std::cout << "\n\nProject name " << ProjectName << "\nNumber of features                                                         " << argv[2] << 
        "\nNumber of LH Trees (multiple of 2 for the banknote dateset)                " << argv[3] << "\nMinimum images(samples) per block allowing split                           " <<
        argv[4] << "\nSensor constant if std. dev. less than                                    " << argv[5]  << std::endl;
    std::cout << "\nResults of testing on TRAINING DATA --  perhaps useful for development \n";
    std::cout << "Mean time to grow a LH Tree         " << trainTime.count() / (double) numberLF_trees << " seconds." << std::endl;
    std::cout << "Mean leaf count of the LH Trees     " << leafCount / (double) numberLF_trees << std::endl;
    std::cout << "Estimated parallel training time                  " << tparallel << " milliseconds. " << std::endl;
    testOnTRAINING_SET(numberLF_trees, apLF_tree);
    testOnTEST_SET(numberLF_trees, apLF_tree);
    testTime(numberLF_trees, apLF_tree);
    free(apLF_tree);
}  // End of main routine and the program run

double reinforcement(uint8_t action, uint8_t label)  // This training is done by the trainer who knows only the system's action and the label
{
       return (action == label) ? 1 : 0; 
}




void loadbanknoteData() {
    std::ifstream file(banknote_data_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << banknote_data_path << std::endl;
        return;
    }

    // Group and store features and labels by label
    std::unordered_map<uint8_t, std::vector<std::vector<double>>> grouped_features;
    std::unordered_map<uint8_t, std::vector<uint8_t>> grouped_labels;

    // Read the data and group it
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;

        // Read the features
        while (std::getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }

        // Read the labels
        double label_raw = features.back(); 
        features.pop_back();               // Remove the label values
        uint8_t label = static_cast<uint8_t>(label_raw);

        // Map label 0 to 1 and label 1 to 2
        if (label == 0) {
            label = 1;
        }
        else if (label == 1) {
            label = 2;
        }

        // Group by label
        grouped_features[label].push_back(features);
        grouped_labels[label].push_back(label);
    }

    file.close();

    // Calculate the mean and standard deviation for each feature
    std::vector<double> feature_means(4, 0.0);
    std::vector<double> feature_stds(4, 0.0);

    // Calculate the mean
    size_t total_samples = 0;
    for (const auto& pair : grouped_features) {
        for (const auto& features : pair.second) {
            for (size_t i = 0; i < features.size(); ++i) {
                feature_means[i] += features[i];
            }
            total_samples++;
        }
    }

    for (size_t i = 0; i < feature_means.size(); ++i) {
        feature_means[i] /= total_samples;
    }

    // Calculate the standard deviation
    for (const auto& pair : grouped_features) {
        for (const auto& features : pair.second) {
            for (size_t i = 0; i < features.size(); ++i) {
                feature_stds[i] += std::pow(features[i] - feature_means[i], 2);
            }
        }
    }

    for (size_t i = 0; i < feature_stds.size(); ++i) {
        feature_stds[i] = std::sqrt(feature_stds[i] / total_samples);
    }

    // Normalize each feature
    for (auto& pair : grouped_features) {
        auto& features = pair.second;

        for (auto& feature_set : features) {
            for (size_t i = 0; i < feature_set.size(); ++i) {
                if (feature_stds[i] != 0) {
                    feature_set[i] = (feature_set[i] - feature_means[i]) / feature_stds[i];
                }
            }
        }
    }

    // Randomly shuffle the data within each label group and split it proportionally
    int random_seed = 33; 
    std::mt19937 g(random_seed);

    for (auto& pair : grouped_features) {
        uint8_t label = pair.first;
        auto& features = pair.second;
        auto& labels = grouped_labels[label];

        // Shuffle the data
        std::vector<size_t> indices(features.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), g);

        // Split the data proportionally
        size_t train_size = static_cast<size_t>(features.size() * 0.8);
        for (size_t i = 0; i < train_size; ++i) {
            train_features.push_back(features[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        }
        for (size_t i = train_size; i < indices.size(); ++i) {
            test_features.push_back(features[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }

    std::cout << "Loaded " << train_features.size() << " training samples and "
        << test_features.size() << " testing samples from banknote dataset." << std::endl;
    std::cout << "Loaded " << train_labels.size() << " training samples and "
        << test_labels.size() << " testing samples from banknote dataset." << std::endl;

    std::vector<uint8_t> first_100(test_labels.begin(), test_labels.begin() + 100);
    std::vector<std::vector<double>> first_100_train(test_features.begin(), test_features.begin() + 100);
    first_100_elements = first_100;
    first_100_elements_train = first_100_train;
}



void testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree)
{
    // Evaluation on the test set using the weighted average of linear functions in leaf SBs of several LF_trees.
    // Using several trees in a weighted average increases the accuracy.
    // Comparing groups of trees with different actions allows the system
    // to choose the action with highest probability of being correct. 
    uint32_t numberoftestimages = 275;
    std::vector<std::vector<double>> pcurrentImage;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<size_t> badRecognitions{};
    P_right.assign(10, 0);
    long goodDecisions = 0;
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        pcurrentImage = test_features;
        label = test_labels[imageNo];
        double P_rightTree = 0; // The probability thata given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for(uint8_t action = 0; action < 3; ++action)
        {
            count = 0;
            accuP_right = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % 2+1 == action)
                {
                   P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage[imageNo], wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right;
            //P_rightAction = accuP_right / accu_weights;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)   // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
                badRecognitions.push_back(imageNo);
            }
        }
    } // Loop  over test images
    double ProbabilityOfCorrect = (double)goodDecisions / numberoftestimages;
    if(lowSumWeightsCount > 0) std::cout << "Low sum weights count " << lowSumWeightsCount << "." << std::endl;
    std::cout << "\nProbability of a correct decision on TESTING DATA                 " << ProbabilityOfCorrect * 100.0 << " percent " << std::endl;
    std::cout << "goodDecisions" << goodDecisions  << std::endl;
} // End of testOnTEST_SET

void testTime(int numberLF_trees, cLF_tree** apLF_tree)
{
    // Evaluation on the test set using the weighted average of linear functions in leaf SBs of several LF_trees.
    // Using several trees in a weighted average increases the accuracy.
    // Comparing groups of trees with different actions allows the system
    // to choose the action with highest probability of being correct. 
    uint32_t numberoftestimages = 100;
    std::vector<std::vector<double>> pcurrentImage;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<size_t> badRecognitions{};
    P_right.assign(10, 0);
    long goodDecisions = 0;
    auto start_test_set = std::chrono::steady_clock::now();
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        pcurrentImage = first_100_elements_train;
        label = test_labels[imageNo];
        double P_rightTree = 0; // The probability thata given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for (uint8_t action = 0; action < 3; ++action)
        {
            count = 0;
            accuP_right = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % 2 + 1 == action)
                {
                    P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage[imageNo], wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right;
            //P_rightAction = accuP_right / accu_weights;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)   // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
                badRecognitions.push_back(imageNo);
            }
        }
    } // Loop  over test images
    auto test_finished = std::chrono::steady_clock::now();
    auto elapsed_test = std::chrono::duration<double, std::milli>(test_finished - start_test_set);
    std::cout << "Mean time required to classify an image           " << elapsed_test.count() / 100.0 << " milliseconds. " << std::endl;


} // End of testTime


void testOnTRAINING_SET(int numberLF_trees, cLF_tree** apLF_tree)
{
    // Evaluation on the training set using the weighted average of linear functions in leaf SBs of several LF_trees.
    // Using several trees in a weighted average increases the accuracy.
    // Comparing groups of trees with different actions allows the system
    // to choose the action with highest probability of being correct. 
    uint32_t numberoftestimages = 1097;
    //mnist_image_t* pcurrentImage = nullptr;
    std::vector<std::vector<double>> pcurrentImage;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<int> confusion;
    confusion.assign(100, 0);
    int countConfusion = 0;
    P_right.assign(10, 0);
    long goodDecisions = 0;
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        //pcurrentImage = train_images + imageNo;
        pcurrentImage = train_features;
        label = train_labels[imageNo];
        double P_rightTree = 0; // The probability thata given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for (uint8_t action = 0; action < 10; ++action)
        {
            count = 0;
            accuP_right = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % 2+1 == action)
                {
                    P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage[imageNo], wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right;
            //P_rightAction = accuP_right / accu_weights;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)  // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
               // std::cout << "Bad training action " << (int) chosenAction << " on numeral " << (int) label << std::endl;
                ++confusion[(int)chosenAction * 10 + label];
                ++countConfusion;
            }
        }
    } // Loop  over training images
    double ProbabilityOfCorrect = (double)goodDecisions / 1097.0;
    if (lowSumWeightsCount > 0) std::cout << "Low sum weights count " << lowSumWeightsCount << "." << std::endl;
    std::cout << "\nProbability of a correct decision on TRAINING DATA               " << ProbabilityOfCorrect * 100.0 << " percent " << std::endl;
    std::cout << "Total mistakes testing on TRAINING DATA " << countConfusion << " or " << countConfusion / 1097.0 << " % " << std::endl;
 } // End of testOnTRAINING_SET
