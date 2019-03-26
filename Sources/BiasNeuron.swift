//
//  BiasNeuron.swift
//  SwiftSimpleNeuralNetwork
//
//  Created by David Kopec on 3/13/18.
//  Copyright © 2018-2019 Oak Snow Consulting. All rights reserved.
//

import Foundation

class BiasNeuron: Neuron {
    // weights are dummies, so other algorithms don't have to change
    init(weights: [Double]) {
        super.init(weights: weights, activationFunction: { _ in return 0.0 }, derivativeActivationFunction: { _ in return 0.0 })
    }
    override func output(inputs: [Double]) -> Double {
        return 1.0
    }
}
