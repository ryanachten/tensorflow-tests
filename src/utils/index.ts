export const logData = (length: number, max: number) => {
  const data = [];
  for (let i = 1; i < length; i++) {
    data.push({
      x: i,
      y: Math.log(i) * max
    });
  }
  return data;
};

export const expData = (length: number, max: number) => {
  const data = [];
  for (let i = 1; i < length; i++) {
    data.push({
      x: i,
      y: Math.exp(i) * max
    });
  }
  return data;
};

export const randomData = (length: number, max: number) => {
  const data = [];
  for (let i = 0; i < length; i++) {
    data.push({
      x: i,
      y: Math.random() * max
    });
  }
  return data;
};
