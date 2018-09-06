//
//  Neuron.swift
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


/// An individual node in a layer
class Neuron:Codable{
    var weights: [Double]
    var activationFunction: (Double) -> Double
    var derivativeActivationFunction: (Double) -> Double
    var inputCache: Double = 0.0
    var delta: Double = 0.0
    var learningRate: Double
    
    init(weights: [Double], activationFunction: @escaping (Double) -> Double, derivativeActivationFunction: @escaping (Double) -> Double, learningRate: Double = 0.25) {
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

    // CODABLE BEGIN
    //  guff for saving / restoring
    private enum CodingKeys: CodingKey {
        case weights
        case activationFunction
        case derivativeActivationFunction
        case inputCache
        case delta
        case learningRate
    }
    
    required convenience init(from decoder: Decoder) throws
    {
        self.init(weights: [0], activationFunction: { _ in return 0.0 }, derivativeActivationFunction: { _ in return 0.0 })
        let container = try decoder.container(keyedBy: CodingKeys.self);
        weights = try container.decode([Double].self, forKey: .weights);
        activationFunction       = { _ in return 0.0 }//  try container.decode(func.self, forKey: .activationFunction)
        derivativeActivationFunction       =  { _ in return 0.0 }//try container.decode(Double.self, forKey: .derivativeActivationFunction)
        inputCache  = try container.decode(Double.self, forKey: .inputCache)
        delta  = try container.decode(Double.self, forKey: .delta)
        learningRate  = try container.decode(Double.self, forKey: .learningRate)
    }
    
    func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(weights, forKey: .weights)
        try container.encode(inputCache, forKey: .inputCache)
        try container.encode(delta, forKey: .delta)
         try container.encode(learningRate, forKey: .learningRate)
        // encoding escape functions ?? activationFunction /derivativeActivationFunction
    }
    // END
}
