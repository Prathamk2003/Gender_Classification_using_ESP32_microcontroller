// # Course: IoT and Edge Computing
// # This code is based on the Lab8a_CarsClassify template and is tailored for the Gender classification model

#include <EloquentTinyML.h>
#include "Gender_Classification.h"  // Header file containing the Gender classification model

#define NUMBER_OF_INPUTS 7       // Define the number of model input features
#define NUMBER_OF_OUTPUTS 1       // Define the number of model output classes
#define TENSOR_ARENA_SIZE 16 * 1024  // Increase tensor arena size for the model

// Initialize the model with defined input, output, and tensor arena size
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

float fResult[NUMBER_OF_OUTPUTS] = {0};  // Array to store prediction results

void setup() {
    Serial.begin(115200);
    if (ml.begin(Gender_Classification)) {
        Serial.println("Model loaded and ready for inference");
    } else {
        Serial.println("Failed to load model.");
    }
}

// Function to display prediction output
void displayOutput(float *fResult) {
    Serial.print("Prediction: ");
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        if (isnan(fResult[i])) {
            Serial.print("NaN ");
        } else {
          Serial.print(fResult[i]);
        }
    }
    Serial.println();
}

void loop() {
    // Example input values for gender classification
    float input1[NUMBER_OF_INPUTS] = {32.0f,175.0f,70.0f,15.0f,3.0f,1.0f,1.0f}; //Output Should be 1


    float input2[NUMBER_OF_INPUTS] = {-1.0f,-170.0f,-1.0f,-29.0f,-1.0f,-6.0f,-131.0f}; //Output should be 0

    // Predict and display results for input1
    if (ml.predict(input1, fResult)) {
        Serial.println("\nThe output value for input1 is:");
        displayOutput(fResult);
    } else {
        Serial.println("Prediction failed for input1.");
    }

    // Predict and display results for input2
    if (ml.predict(input2, fResult)) {
        Serial.println("\nThe output value for input2 is:");
        displayOutput(fResult);
    } else {
        Serial.println("Prediction failed for input2.");
    }

    delay(5000); // Delay for 5 seconds
}
