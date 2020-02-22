import React from "react";
import json from "./data/bench.json";
import { randomData, logData } from "./utils";
import DigitRecogniser from "./components/DigitRecogniser";
const data = logData(18, 1000);

const App = () => (
  <div>
    <DigitRecogniser />
  </div>
);
export default App;
