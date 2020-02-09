import React from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { Tensor, Tensor1D, Rank, Scalar } from "@tensorflow/tfjs";

type Props = {};

type State = {};

type Car = {
  mpg: number;
  horsepower: number;
};

// Goal here is to train a model that will take one number, Horsepower and learn to predict one number
// i.e. one-to-one mapping

class MilesPerGallon extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.run();
  }

  async getData() {
    const carsDataReq = await fetch(
      "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
    );
    const carsData = await carsDataReq.json();
    const cleaned = carsData
      .map((car: any) => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
      }))
      .filter((car: Car) => car.mpg != null && car.horsepower != null);

    return cleaned;
  }

  createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single hidden layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1 }));

    return model;
  }

  convertToTensor(data: Car[]) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);

      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.horsepower);
      const labels = data.map(d => d.mpg);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor
        .sub(labelMin)
        .div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin
      };
    });
  }

  async run() {
    // Load and plot the original input data that we are going to train on.
    const data = await this.getData();
    const values = data.map((d: Car) => ({
      x: d.horsepower,
      y: d.mpg
    }));

    tfvis.render.scatterplot(
      { name: "Horsepower v MPG" },
      { values },
      {
        xLabel: "Horsepower",
        yLabel: "MPG",
        height: 300
      }
    );

    const model = this.createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, model);
  }

  render() {
    return <div>MilesPerGallon</div>;
  }
}

export default MilesPerGallon;
