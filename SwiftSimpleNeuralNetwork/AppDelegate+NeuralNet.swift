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

extension AppDelegate {

    //read the mnist training images (48 MBs, 60000 images, 28x28 grayscale)
    func loadTrainingImages() {
        
        if let path = Bundle.main.path(forResource: "train-images", ofType: "idx3-ubyte"), let stream = InputStream(fileAtPath: path) {
            stream.open()
            defer { stream.close() }
            var dummy: [Byte] = [Byte](repeating: 0, count: 16)
            // skip over header
            stream.read(&dummy, maxLength: 16)
            var i = 0
            var image: [Byte] = [Byte](repeating: 0, count: 784) // 28 x 28 = 784
            while stream.read(&image, maxLength: 784) == 784 {
//                if (i > 10000){break}
                trainingImages.append(image)
                i += 1
                
            }
        }
        
    }
    
    //read the mnist training images (60000 labels, 1 byte each)
    func loadTrainingLabels() {
        
        if let path = Bundle.main.path(forResource: "train-labels", ofType: "idx1-ubyte"), let stream = InputStream(fileAtPath: path) {
            stream.open()
            defer { stream.close() }
            var dummy: [Byte] = [Byte](repeating: 0, count: 8)
            // skip over header
            stream.read(&dummy, maxLength: 8)
            var label: Byte = 0 // 1 byte 0-10
            while stream.read(&label, maxLength: 1) == 1 {
                trainingLabels.append(label)
            }
        }
        
    }
    
    //read the mnist testing images
    func loadTestingImages() {
        
        if let path = Bundle.main.path(forResource: "t10k-images", ofType: "idx3-ubyte"), let stream = InputStream(fileAtPath: path) {
            stream.open()
            defer { stream.close() }
            var dummy: [Byte] = [Byte](repeating: 0, count: 16)
            // skip over header
            stream.read(&dummy, maxLength: 16)
            var image: [Byte] = [Byte](repeating: 0, count: 784) // 28 x 28 = 784
            while stream.read(&image, maxLength: 784) == 784 {
                testingImages.append(image)
            }
        }
        
    }
    
    //read the mnist testing labels
    func loadTestingLabels() {
        
        if let path = Bundle.main.path(forResource: "t10k-labels", ofType: "idx1-ubyte"), let stream = InputStream(fileAtPath: path) {
            stream.open()
            defer { stream.close() }
            var dummy: [Byte] = [Byte](repeating: 0, count: 8)
            // skip over header
            stream.read(&dummy, maxLength: 8)
            var label: Byte = 0 // 1 byte 0-10
            while stream.read(&label, maxLength: 1) == 1 {
                testingLabels.append(label)
            }
        }
        
    }
    
    // based on https://stackoverflow.com/a/34677134/281461
    func imageFromGrayscaleBytes(pixelValues: [Byte]) -> NSImage {
        var pixelValues = pixelValues
        let width = 28
        let height = 28
        let bitsPerComponent = 8
        let bytesPerPixel = 1
        let bitsPerPixel = bytesPerPixel * bitsPerComponent
        let bytesPerRow = bytesPerPixel * width
        let totalBytes = height * bytesPerRow
        let providerRef = CGDataProvider(dataInfo: nil, data: &pixelValues, size: totalBytes) { (info, data, size) in
            return
        }
        
        let colorSpaceRef = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo()
        let imageRef = CGImage(width: width,
                               height: height,
                               bitsPerComponent: bitsPerComponent,
                               bitsPerPixel: bitsPerPixel,
                               bytesPerRow: bytesPerRow,
                               space: colorSpaceRef,
                               bitmapInfo: bitmapInfo,
                               provider: providerRef!,
                               decode: nil,
                               shouldInterpolate: false,
                               intent: CGColorRenderingIntent.defaultIntent)
        return NSImage(cgImage: imageRef!, size: NSSize(width: 28, height: 28))
    }

    func showTrainingImage(index: Int) {
        characterImageView.image = imageFromGrayscaleBytes(pixelValues: trainingImages[index])
        currentCharacterIndexLabel.stringValue = "Index: \(index)"
        currentCharacterLabel.stringValue = "Label: \(trainingLabels[index])"
    }
    
    func showTestingImage(index: Int) {
        testingImageView.image = imageFromGrayscaleBytes(pixelValues: testingImages[index])
        testingCharacterIndexLabel.stringValue = "Index: \(index)"
        testingCharacterLabel.stringValue = "Label: \(testingLabels[index])"
    }

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Insert code here to initialize your application
        //print(randomWeights(number: 10))
        loadTrainingImages()
        loadTrainingLabels()
        loadTestingImages()
        loadTestingLabels()
        // show the first one
        showTrainingImage(index: currentTrainingImageShown)
        showTestingImage(index: currentTestingImageShown)
    }


    
    func train(){
        // original images are made of bytes from 0-255, we need these intensity values as Doubles
        let imageData = trainingImages.map{ return $0.map{ return Double($0) / 255 }}
        let expecteds: [[Double]] = trainingLabels.map{
            switch $0 {
            case 0:
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            case 1:
                return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            case 2:
                return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            case 3:
                return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            case 4:
                return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            case 5:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            case 6:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            case 7:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            case 8:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            default:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
        // batches
        //        for _ in 0..<100 {
        //            network.train(inputs: Array<Array<Double>>(imageData.prefix(50)), expecteds: Array<Array<Double>>(expecteds.prefix(50)), printError: true)
        //        }
        // all
        DispatchQueue.main.async { [unowned self] in
            self.trainButton.isEnabled = false
            self.testAllButton.isEnabled = false
            self.testButton.isEnabled = false
            self.trainProgress.doubleValue = 0.0
        }
        DispatchQueue.global(qos: DispatchQoS.QoSClass.userInitiated).async { [unowned self, numBatches = numBatches] in
            for i in 1...numBatches {
                self.network.train(inputs: imageData, expecteds: expecteds, printError: false)
                print("Finished batch \(i) of \(numBatches)")
                DispatchQueue.main.async { [unowned self] in
                    self.trainProgress.doubleValue = Double(i)
                }
            }
            print("Done Training")
            DispatchQueue.main.async { [unowned self] in
                self.trainButton.isEnabled = true
                self.testAllButton.isEnabled = true
                self.testButton.isEnabled = true
            }
        }
    }
    
 
}

