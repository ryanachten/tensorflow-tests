import React from "react";
import * as tf from "@tensorflow/tfjs";
import { Sequential } from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";

type Props = {};

type State = {};

class LayersTest extends React.Component<Props, State> {
  hidden: Layer;
  model: Sequential;
  output: Layer;

  constructor(props: Props) {
    super(props);

    // Sequential neural network model
    this.model = tf.sequential();

    // Layer setup
    this.hidden = tf.layers.dense({
      inputShape: [2],
      units: 4,
      activation: "sigmoid"
    });
    this.output = tf.layers.dense({
      units: 3,
      activation: "sigmoid"
    });

    // Add the layers to the sequential model
    this.model.add(this.hidden);
    this.model.add(this.output);
  }

  render() {
    return <div>Layers</div>;
  }
}

export default LayersTest;
