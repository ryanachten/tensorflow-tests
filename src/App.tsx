import React from "react";
import LinearRegression from "./components/LinearRegression";
import PolynomialRegression from "./components/PolynomialRegression";
import json from "./data/bench.json";
import { randomData, logData } from "./utils";
import LayersTest from "./components/Layers";
import Fibonacci from "./components/Fibonacci";
const data = logData(18, 1000);

const App = () => (
  <div>
    <Fibonacci />
    {/* <LinearRegression
      animate={false}
      data={data}
      epochs={150}
      learningRate={0.005}
    />
    <PolynomialRegression
      animate={false}
      data={data}
      epochs={150}
      learningRate={0.005}
    /> */}
  </div>
);
export default App;
