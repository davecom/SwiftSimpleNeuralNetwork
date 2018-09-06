//
//  AppDelegate.swift
//  SwiftSimpleNeuralNetwork
//
//  Copyright 2018 David Kopec
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

//MNIST data set curtosy http://yann.lecun.com/exdb/mnist/

import Cocoa

typealias Byte = UInt8

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {

    @IBOutlet weak var window: NSWindow!
    @IBOutlet weak var characterImageView: NSImageView!
    @IBOutlet weak var currentCharacterIndexLabel: NSTextField!
    @IBOutlet weak var currentCharacterLabel: NSTextField!
    @IBOutlet weak var testingImageView: NSImageView!
    @IBOutlet weak var testingCharacterIndexLabel: NSTextField!
    @IBOutlet weak var testingCharacterLabel: NSTextField!
    @IBOutlet weak var singleTestPredictionLabel: NSTextField!
    @IBOutlet weak var allTestAccuracyLabel: NSTextField!
    @IBOutlet weak var trainProgress: NSProgressIndicator!
    @IBOutlet weak var batchLabel: NSTextField!
    
    @IBOutlet weak var testButton: NSButton!
    @IBOutlet weak var testAllButton: NSButton!
    @IBOutlet weak var trainButton: NSButton!
    
    var trainingImages = [[Byte]]()
    var trainingLabels = [Byte]()
    var testingImages = [[Byte]]()
    var testingLabels = [Byte]()
    var currentTrainingImageShown: Int = 0
    var currentTestingImageShown: Int = 0
    var numBatches: Int = 1
    let network: Network = Network(layerStructure: [784,20,10], activationFunction: sigmoid, derivativeActivationFunction: derivativeSigmoid, learningRate: 0.006, hasBias: true)
    
 

  
    @IBAction func nextTrainingImage(sender: AnyObject) {
        currentTrainingImageShown = (currentTrainingImageShown + 1) % trainingImages.count
        showTrainingImage(index: currentTrainingImageShown)
    }

    @IBAction func previousTrainingImage(sender: AnyObject) {
        currentTrainingImageShown -= 1
        if currentTrainingImageShown < 0 { currentTrainingImageShown = trainingImages.count - 1 }
        showTrainingImage(index: currentTrainingImageShown)
    }
    
    @IBAction func nextTestingImage(sender: AnyObject) {
        currentTestingImageShown = (currentTestingImageShown + 1) % testingImages.count
        showTestingImage(index: currentTestingImageShown)
    }
    
    @IBAction func previousTestingImage(sender: AnyObject) {
        currentTestingImageShown -= 1
        if currentTestingImageShown < 0 { currentTestingImageShown = testingImages.count - 1 }
        showTestingImage(index: currentTestingImageShown)
    }
    
    @IBAction func train(sender: AnyObject) {
       self.train()
    }
    
    // helper function finds the highest in an Array of Double and returns its index
    func interpretOutput(output: [Double]) -> Int {
        return output.index(of: output.max()!)!
    }
    
    @IBAction func batchSizeChanged(sender: NSStepper) {
        numBatches = sender.integerValue
        trainProgress.maxValue = sender.doubleValue
        batchLabel.stringValue = "\(numBatches)"
    }
    
    @IBAction func testCurrent(sender: AnyObject) {
        let imageData = testingImages[currentTestingImageShown].map{ return Double($0)}
        let output = network.outputs(input: imageData)
        print(output)
        let result = interpretOutput(output: output)
        singleTestPredictionLabel.stringValue = "Prediction: \(result)"
    }
    
    @IBAction func testAll(sender: AnyObject) {
        let imageData = testingImages.map{ return $0.map{ return Double($0) / 255 }}
        let expecteds: [Int] = testingLabels.map{ Int($0) }
        let (_, _, percentage) = network.validate(inputs: imageData, expecteds: expecteds, interpretOutput: interpretOutput)
        allTestAccuracyLabel.stringValue = "Accuracy: \(percentage * 100)%"
    }
}

