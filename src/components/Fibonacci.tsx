import React from "react";
import * as tf from "@tensorflow/tfjs";
import {
  Sequential,
  SGDOptimizer,
  Tensor,
  Tensor1D,
  Rank
} from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";

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
    function fibonacci(num: number) {
      var a = 1,
        b = 0,
        temp;
      var seq = [];

      while (num > 0) {
        temp = a;
        a = a + b;
        b = temp;
        seq.push(b);
        num--;
      }

      return seq;
    }

    const fibs = fibonacci(100);

    const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1));
    const ys = tf.tensor1d(fibs.slice(1));

    const xmin = xs.min();
    const xmax = xs.max();
    const xrange = xmax.sub(xmin);

    function norm(x: Tensor1D) {
      return x.sub(xmin).div(xrange);
    }

    const xsNorm = norm(xs);
    const ysNorm = norm(ys);

    // Building our model
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));

    a.print();
    b.print();

    function predict(x: Tensor<Rank>): Tensor<Rank.R0> {
      return tf.tidy(() => {
        return a.mul(x).add(b);
      });
    }

    // Training

    function loss(
      predictions: Tensor<Rank>,
      labels: Tensor<Rank>
    ): Tensor<Rank.R0> {
      return predictions
        .sub(labels)
        .square()
        .mean();
    }

    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    const numIterations = 10000;
    const errors: any[] = [];

    for (let iter = 0; iter < numIterations; iter++) {
      optimizer.minimize(() => {
        const predsYs = predict(xsNorm);
        const e = loss(predsYs, ysNorm);
        errors.push(e.dataSync());
        return e;
      });
    }

    // Making predictions

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
