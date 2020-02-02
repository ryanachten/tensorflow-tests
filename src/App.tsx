import React from "react";
import LinearRegression from "./LinearRegression";
import PolynomialRegression from "./PolynomialRegression";

const App = () => (
  <div>
    <LinearRegression learningRate={0.005} />
    <PolynomialRegression learningRate={0.005} />
  </div>
);
export default App;
