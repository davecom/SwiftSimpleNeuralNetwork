//
//  IrisTest.swift
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


// Iris data set courtsey of:
// Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

import XCTest
import Foundation

/// Tests against the classic iris data set.
/// Uses the entire data set both for training
/// and for validation (yes, bad probably).
class IrisTest: XCTestCase {

    var network: Network = Network(layerStructure: [4,20,3], learningRate: 0.3, hasBias: true)
    var irisParameters: [[Double]] = [[Double]]()
    var irisClassifications: [[Double]] = [[Double]]()
    var irisSpecies: [String] = [String]()
    
    func parseIrisCSV() {
        let myBundle = Bundle.init(for: IrisTest.self)
        let urlpath = myBundle.path(forResource: "iris", ofType: "csv")
        let url = URL(fileURLWithPath: urlpath!)
        let csv = try! String.init(contentsOf: url)
        let lines = csv.components(separatedBy: "\n")
        
        let shuffledLines = lines.shuffled
        for line in shuffledLines {
            if line == "" { continue }
            let items = line.components(separatedBy: ",")
            let parameters = items[0...3].map{ Double($0)! }
            irisParameters.append(parameters)
            let species = items[4]
            if species == "Iris-setosa" {
                irisClassifications.append([1.0, 0.0, 0.0])
            } else if species == "Iris-versicolor" {
                irisClassifications.append([0.0, 1.0, 0.0])
            } else {
                irisClassifications.append([0.0, 0.0, 1.0])
            }
            irisSpecies.append(species)
        }
        normalizeByColumnMax(dataset: &irisParameters)
    }
    
    func interpretOutput(output: [Double]) -> String {
        if output.max()! == output[0] {
            return "Iris-setosa"
        } else if output.max()! == output[1] {
            return "Iris-versicolor"
        } else {
            return "Iris-virginica"
        }
    }
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
        parseIrisCSV()
        // train over entire data set 10 times
        for _ in 0..<1000 {
            network.train(inputs: irisParameters, expecteds: irisClassifications, printError: false)
        }
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testSamples() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        let results = network.validate(inputs: irisParameters, expecteds: irisSpecies, interpretOutput: interpretOutput)
        print("\(results.correct) correct of \(results.total) = \(results.percentage * 100)%")
        XCTAssertEqual(results.percentage, 1.00, accuracy: 0.05, "Did not come within a 95% confidence interval")
    }

}
