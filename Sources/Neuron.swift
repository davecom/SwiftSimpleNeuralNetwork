//
//  Neuron.swift
//  SwiftSimpleNeuralNetwork
//
//  Created by David Kopec on 6/11/16.
//  Copyright © 2016 Oak Snow Consulting. All rights reserved.
//

/// An individual node in a layer
class Neuron {
    var weights: [Double]
    var activationFunction: (Double) -> Double
    var derivativeActivationFunction: (Double) -> Double
    var inputCache: Double = 0.0
    var delta: Double = 0.0
    var learningRate: Double
    
    init(weights: [Double], activationFunction: (Double) -> Double, derivativeActivationFunction: (Double) -> Double, learningRate: Double = 0.25) {
        self.weights = weights
        self.activationFunction = activationFunction
        self.derivativeActivationFunction = derivativeActivationFunction
        self.learningRate = learningRate
    }
    
    /// The output that will be going to the next layer
    /// or the final output if this is an output layer
    func output(inputs: [Double]) -> Double {
        inputCache = dotProduct(inputs, weights)
        return activationFunction(inputCache)
    }
    
}
