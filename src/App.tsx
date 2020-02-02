import React from "react";
import * as tf from "@tensorflow/tfjs";
import {
  VictoryChart,
  VictoryScatter,
  VictoryTheme,
  VictoryLine
} from "victory";
import data from "./data.json";
import { Scalar, Tensor, Rank, SGDOptimizer } from "@tensorflow/tfjs";

type Props = {};

type State = {
  cycles: number;
};

class LinearRegression extends React.Component<Props, State> {
  data: { x: number; y: number; id: string }[];
  ys: number[];
  xs: number[];
  b: Scalar;
  m: Scalar;
  learningRate: number;
  optimizer: SGDOptimizer;
  interval: number | undefined;

  constructor(props: Props) {
    super(props);
    this.data = data.map(({ id, x, y }) => ({ id, x, y }));
    this.ys = data.map(({ y }) => y);
    this.xs = data.map(({ x }) => x);
    // Rate at which the values will be adjusted
    this.learningRate = 0.005;
    this.optimizer = tf.train.sgd(this.learningRate);
    this.train = this.train.bind(this);

    // Initially set m (slope) and b (y-intercept) as a random value
    // and adjust them later
    this.b = tf.variable(tf.scalar(Math.random()));
    this.m = tf.variable(tf.scalar(Math.random()));

    this.state = {
      cycles: 0
    };
  }

  componentDidMount() {
    this.interval = window.setInterval(() => {
      if (this.state.cycles > 150) {
        return window.clearInterval(this.interval);
      }
      tf.tidy(() => this.train());
    }, 1000);
  }

  predict(x: Tensor<Rank.R1>): Tensor<Rank> {
    const prediction = tf.tidy(() => this.m.mul(x).add(this.b));
    this.setState(prevState => ({
      cycles: prevState.cycles + 1
    }));
    return prediction;
  }

  // Predictions are produced via the prediction method
  // Labels are the y vals from the data set
  loss(predictions: Tensor<Rank.R1>, labels: Tensor<Rank.R1>): Scalar {
    // Loss function using the square mean error approach
    return predictions
      .sub(labels)
      .square()
      .mean();
  }

  train() {
    this.optimizer.minimize(() => {
      const predsYs = this.predict(tf.tensor1d(this.xs)) as Tensor<Rank.R1>;
      const tensorYs = tf.tensor1d(this.ys);
      const stepLoss = this.loss(predsYs, tensorYs);
      tensorYs.dispose();
      return stepLoss;
    });
  }

  render() {
    const xMax = Math.max(...this.xs) + 10;
    const line = [
      {
        x: 0,
        y: this.b.dataSync()[0]
      },
      {
        x: xMax,
        y: xMax * this.m.dataSync()[0] + this.b.dataSync()[0]
      }
    ];

    return (
      <VictoryChart animate theme={VictoryTheme.material}>
        <VictoryLine data={line} />
        <VictoryScatter data={this.data} />
      </VictoryChart>
    );
  }
}

export default LinearRegression;
