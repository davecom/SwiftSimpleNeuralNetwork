//
//  SwiftSimpleNeuralNetworkTests.swift
//  SwiftSimpleNeuralNetworkTests
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


import XCTest

class SwiftSimpleNeuralNetworkTests: XCTestCase {
    
    var network: Network = Network(layerStructure: [1,8,1], learningRate: 8.0)
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
        let xos = randomNums(number: 10000, limit: 500)
        let ys = xos.map{[(0.5 * sin(10 * ($0 / 1000)) + 0.5)]}
        let xs = xos.map{[$0 / 1000]}
        network.train(inputs: xs, expecteds: ys, printError: true)
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testSin() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        let xos = randomNums(number: 1000, limit: 300)
        let ys = xos.map{ (0.5 * sin(10 * ($0 / 600)) + 0.5) }
        let xs = xos.map{[$0 / 600]}
        let results = network.validate(inputs: xs, expecteds: ys, accuracy: 0.05)
        print("\(results.correct) correct of \(results.total) = \(results.percentage * 100)%")
        XCTAssertEqualWithAccuracy(results.percentage, 1.00, accuracy: 0.05, "Did not come within a 95% confidence interval")
    }
    
}
