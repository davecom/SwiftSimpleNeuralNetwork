//
//  Network.swift
//  SwiftSimpleNeuralNetwork
//
//  Copyright 2016 David Kopec
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import Foundation // for sqrt

class Network {
    var layers: [Layer]
    
    init(layerStructure:[Int], activationFunction: (Double) -> Double = sigmoid, derivativeActivationFunction: (Double) -> Double = derivativeSigmoid, learningRate: Double = 0.25) {
        if (layerStructure.count < 3) {
            print("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        }
        layers = [Layer]()
        // input layer
        layers.append(Layer(numNeurons: layerStructure[0], activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate))
        
        // hidden layers and output layer
        for x in layerStructure.enumerated() where x.offset != 0 {
            layers.append(Layer(previousLayer: layers[x.offset - 1], numNeurons: x.element, activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate))
        }
    }
    
    func outputs(input: [Double]) -> [Double] {
        return layers.reduce(input) { $1.outputs(inputs: $0) }
    }
    
    func backPropagate(expected: [Double]) {
        //calculate delta for output layer neurons
        layers.last?.calculateDeltasForOutputLayer(expected: expected)
        //calculate delta for prior layers
        for l in 1..<layers.count - 1 {
            layers[l].calculateDeltasForHiddenLayer(nextLayer: layers[l + 1])
        }
    }
    
    func updateWeights() {
        for layer in layers {
            for neuron in layer.neurons {
                for w in 0..<neuron.weights.count {
                    neuron.weights[w] = neuron.weights[w] + (neuron.learningRate * (layer.previousLayer?.outputCache[w])!  * neuron.delta)
                }
            }
        }
    }
    
    func train(inputs:[[Double]], expecteds:[[Double]], printError:Bool = false, threshold:Double? = nil) {
        for (location, xs) in inputs.enumerated() {
            let ys = expecteds[location]
            let outs = outputs(input: xs)
            if (printError) {
                let diff = sub(x: outs, y: ys)
                let error = sqrt(sum(x: mul(x: diff, y: diff)))
                print("\(error) error in run \(location)")
            }
            backPropagate(expected: ys)
            updateWeights()
        }
    }
    
    // for generalized results that require classification
    func validate<T: Equatable>(inputs:[[Double]], expecteds:[T], interpretOutput: ([Double]) -> T) -> (correct: Int, total: Int, percentage: Double) {
        var correct = 0
        for (input, expected) in zip(inputs, expecteds) {
            let result = interpretOutput(outputs(input: input))
            if result == expected {
                correct += 1
            }
        }
        let percentage = Double(correct) / Double(inputs.count)
        return (correct, inputs.count, percentage)
    }
    
    // for when result is a single neuron
    func validate(inputs:[[Double]], expecteds:[Double], accuracy: Double) -> (correct: Int, total: Int, percentage: Double) {
        var correct = 0
        for (input, expected) in zip(inputs, expecteds) {
            let result = outputs(input: input)[0]
            if abs(expected - result) < accuracy {
                correct += 1
            }
        }
        let percentage = Double(correct) / Double(inputs.count)
        return (correct, inputs.count, percentage)
    }
}
