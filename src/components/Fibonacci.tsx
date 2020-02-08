import React from "react";
import * as tf from "@tensorflow/tfjs";
import { Sequential, SGDOptimizer, model, Tensor } from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";
import json from "../data/bench.json";

type Props = {};

type State = {};

class Fibonacci extends React.Component<Props, State> {
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
    const fibs = this.fibonacci(100);
    const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1));
    const ys = tf.tensor1d(fibs.slice(1));

    // Scale down data to work better with tensorflow
    let xsNorm = this.normalise(xs);
    let ysNorm = this.normalise(ys);

    let a = tf.variable(tf.scalar(Math.random()));
    let b = tf.variable(tf.scalar(Math.random()));

    // Linear regression predictor
    function predict(x: Tensor<tf.Rank>): Tensor<tf.Rank> {
      return tf.tidy(() => {
        return a.mul(x).add(b);
      });
    }

    function loss(
      predictions: Tensor<tf.Rank>,
      labels: Tensor<tf.Rank>
    ): Tensor<tf.Rank.R0> {
      return predictions
        .sub(labels)
        .square()
        .mean();
    }

    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    const numIterations = 10000;
    const errors: any[] = [];
    // Calculate error using predicted values
    for (let i = 0; i < numIterations; i++) {
      optimizer.minimize(() => {
        const predsYs = predict(xsNorm);
        const e = loss(predsYs, ysNorm);
        errors.push(e.dataSync());
        return e;
      });
    }

    console.log(errors[0]);
    console.log(errors[numIterations - 1]);

    const xTest = tf.tensor1d([2, 354224848179262000000]);
    predict(xTest).print();

    a.print();
    b.print();
  }

  fibonacci(range: number) {
    let a = 1,
      b = 0,
      temp;
    const seq = [];
    while (range > 0) {
      temp = a;
      a = a + b;
      b = temp;
      seq.push(b);
      range--;
    }
    return seq;
  }

  normalise(n: Tensor<tf.Rank.R1>) {
    const nMin = n.min();
    const nMax = n.max();
    const nRange = nMax.sub(nMin);
    return n.sub(nMin).div(nRange);
  }

  render() {
    return <div>Fibonacci</div>;
  }
}

export default Fibonacci;
