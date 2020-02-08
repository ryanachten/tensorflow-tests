import React from "react";
import * as tf from "@tensorflow/tfjs";
import { Sequential, SGDOptimizer, model } from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";

type Props = {};

type State = {};

class LayersTest extends React.Component<Props, State> {
  hidden?: Layer;
  learningRate?: number;
  model?: Sequential;
  optimiser?: SGDOptimizer;
  output?: Layer;

  constructor(props: Props) {
    super(props);
    this.initModel();
  }

  async initModel() {
    // Sequential neural network model
    this.model = tf.sequential();

    // Layer setup
    this.hidden = tf.layers.dense({
      inputShape: [2],
      units: 4,
      activation: "sigmoid"
    });
    this.output = tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    });

    // Add the layers to the sequential model
    this.model.add(this.hidden);
    this.model.add(this.output);

    this.learningRate = 0.1;
    this.optimiser = tf.train.sgd(this.learningRate);

    this.model.compile({
      optimizer: this.optimiser,
      loss: tf.losses.meanSquaredError
    });

    // TODO: replace these inputs with real data
    const xs = tf.tensor2d([
      [0, 0],
      [0.5, 0.5],
      [1, 1]
    ]);

    const ys = tf.tensor2d([[1], [0.5], [0]]);

    // Train the model
    // NOTE: Loop here is just to log out the loss - don't use in PROD
    // for (let index = 0; index < 1000; index++) {
    //   const res = await this.model.fit(xs, ys, { epochs: 10, shuffle: true });
    //   console.log("loss", res.history.loss[0]);
    // }

    await this.model.fit(xs, ys, { epochs: 10000, shuffle: true });

    // Outputs can either be returned as a tensor array or a single tensor
    const output = this.model.predict(xs);
    if ("length" in output) {
      output.map(o => o.print());
    } else {
      output.print();
    }
  }

  render() {
    return <div>Layers</div>;
  }
}

export default LayersTest;
