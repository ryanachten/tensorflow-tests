import React from "react";
import * as tf from "@tensorflow/tfjs";
import {
  VictoryChart,
  VictoryScatter,
  VictoryTheme,
  VictoryLine
} from "victory";
import { Tensor, Rank } from "@tensorflow/tfjs";
import LinearRegression from "./LinearRegression";

class PolynomialRegression extends LinearRegression {
  a: number;
  c: number;
  constructor(props: any) {
    super(props);
    this.a = Math.random();
    this.c = Math.random();

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
      this.setState(prevState => ({
        cycles: prevState.cycles + 1
      }));
    }, 1000);
  }

  predict(x: Tensor<Rank.R1>): Tensor<Rank> {
    // Polynomial regression formula: y = ax^2 + bx + c
    const prediction = tf.tidy(() =>
      x
        .square()
        .mul(this.a)
        .add(x.mul(this.b))
        .add(this.c)
    );
    return prediction;
  }

  render() {
    const curveX: number[] = [];
    const xMax = Math.max(...this.xs);
    for (let index = 0; index < xMax; index++) {
      curveX.push(index);
    }

    const ys = tf.tidy(() => this.predict(tf.tensor1d(curveX)));
    const curveY = ys.dataSync();
    ys.dispose();

    const curve: { x: number; y: number }[] = curveX.map(
      (x: number, i: number) => {
        return {
          x,
          y: curveY[i]
        };
      }
    );

    return (
      <VictoryChart animate theme={VictoryTheme.material}>
        <VictoryLine data={curve} />
        <VictoryScatter data={this.data} />
      </VictoryChart>
    );
  }
}

export default PolynomialRegression;
