using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using TensorFlow;

public class MNIST_Classifier : MonoBehaviour {

	public TextAsset graphModel;							// The trained TensorFlow graph
	public Texture2D[] inputTextures;						// Textures to test with
	private int incImg = 0;									// Increments image displayed

	public Material displayMaterial;						// Material the display the MNIST image
	private Texture baseTexture;							// Texture the material started with

	public Text label;										// Label of classification

	private static int img_width = 28;						// Image width
	private static int img_height = 28;						// Image height
	private float[,,,] inputImg = 
		new float[1,img_width,img_height,1]; 				// Input to model

	// Use this for initialization
	void Start () {
		baseTexture = displayMaterial.mainTexture;

		if (inputTextures.Length > 0) {
			Evaluate (inputTextures [0]);
		}
	}
	
	// Update is called once per frame
	void Update () {
		// Change images using the arrow keys
		if (Input.GetKeyDown (KeyCode.RightArrow)) {
			if (++incImg >= inputTextures.Length) {
				incImg = 0;
			}
			Evaluate (inputTextures [incImg]);
		} else if (Input.GetKeyDown (KeyCode.LeftArrow)) {
			if (--incImg < 0) {
				incImg = inputTextures.Length - 1;
			}
			Evaluate (inputTextures [incImg]);
		}
	}

	// Reset the material texture before exiting
	void OnApplicationQuit () {
		displayMaterial.mainTexture = baseTexture;
	}

	// Classify an MNIST image by running the model
	void Evaluate (Texture2D input) {
		
		// Get raw pixel values from texture, format for inputImg array
		for (int i = 0; i < img_width * img_height; i++) {
			inputImg [0, (img_height-1) - (i / (img_width)), i % img_width, 0] = input.GetPixel (i % img_width, i / img_width).r;
		}

		// Apply texture to displayMaterial
		displayMaterial.mainTexture = input;

		// Create the TensorFlow model
		var graph = new TFGraph();
		graph.Import (graphModel.bytes);
		var session = new TFSession (graph);
		var runner = session.GetRunner ();

		// Set up the input tensor and input
		runner.AddInput (graph ["conv2d_1_input"] [0], inputImg);
		// Set up the output tensor
		runner.Fetch (graph ["dense_2/Softmax"] [0]);

		// Run the model
		float[,] recurrent_tensor = runner.Run () [0].GetValue () as float[,];

		// Find the answer the model is most confident in
		float highest_val = 0;
		int highest_ind = -1;
		float sum = 0;
		float currTime = Time.time;

		for (int j = 0; j < 10; j++) {
			float confidence = recurrent_tensor [0, j];
			if (highest_ind > -1) {
				if (recurrent_tensor [0, j] > highest_val) {
					highest_val = confidence;
					highest_ind = j;
				}
			} else {
				highest_val = confidence;
				highest_ind = j;
			}

			// sum should total 1 in the end
			sum += confidence;
		}

		// Display the answer to the screen
		label.text = "Answer: " + highest_ind + "\n Confidence: " + highest_val +
			"\nLatency: " + (Time.time - currTime) * 1000000 + " us";
	}
}
