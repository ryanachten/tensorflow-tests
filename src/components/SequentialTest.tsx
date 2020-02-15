import React from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { Tensor, Rank, Sequential } from "@tensorflow/tfjs";
import json from "../data/bench.json";

type Props = {};

type State = {};

type Exercise = {
  session: number;
  total: number;
};

type NormalizationData = {
  inputs: tf.Tensor<tf.Rank>;
  labels: tf.Tensor<tf.Rank>;
  inputMax: tf.Tensor<tf.Rank>;
  inputMin: tf.Tensor<tf.Rank>;
  labelMax: tf.Tensor<tf.Rank>;
  labelMin: tf.Tensor<tf.Rank>;
};

// Goal here is to train a model that will take one number, Horsepower and learn to predict one number
// i.e. one-to-one mapping

class SequentialTest extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.run();
  }

  async getData() {
    const exercises: Exercise[] = json.map(d => ({
      session: d.x,
      total: d.y
    }));

    return exercises;
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

  convertToTensor(data: Exercise[]) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);

      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.session);
      const labels = data.map(d => d.total);

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
    const values = data.map((d: Exercise) => ({
      x: d.session,
      y: d.total
    }));

    tfvis.render.scatterplot(
      { name: "Total v Session" },
      { values },
      {
        xLabel: "Session",
        yLabel: "Total",
        height: 300
      }
    );

    const model = this.createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, model);

    // Convert the data to a form we can use for training.
    const tensorData = this.convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Train the model
    await this.trainModel(model, inputs, labels);
    console.log("Done Training");

    // Make some predictions using the model and compare them to the
    // original data
    this.testModel(model, data, tensorData);
  }

  async trainModel(
    model: Sequential,
    inputs: Tensor<Rank>,
    labels: Tensor<Rank>
  ) {
    // Prepare the model for training.
    model.compile({
      optimizer: tf.train.sgd(0.05),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"]
    });

    const batchSize = 32;
    const epochs = 100;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] }
      )
    });
  }

  testModel(
    model: Sequential,
    inputData: Exercise[],
    normalizationData: NormalizationData
  ) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));

      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

      if ("length" in preds) {
        console.log("preds is an array and this wont work");
      }

      const unNormPreds =
        "length" in preds
          ? preds[0].mul(labelMax.sub(labelMin)).add(labelMin)
          : preds.mul(labelMax.sub(labelMin)).add(labelMin);

      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
    });

    const originalPoints = inputData.map(d => ({
      x: d.session,
      y: d.total
    }));

    tfvis.render.scatterplot(
      { name: "Model Predictions vs Original Data" },
      {
        values: [originalPoints, predictedPoints],
        series: ["original", "predicted"]
      },
      {
        xLabel: "Session",
        yLabel: "Total",
        height: 300
      }
    );
  }

  render() {
    return <div>SequentialTest</div>;
  }
}

export default SequentialTest;
