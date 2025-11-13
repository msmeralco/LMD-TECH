import React, { useState } from 'react';

type SampleTestProps = {
  initialValue?: string;
};

export const SampleTest: React.FC<SampleTestProps> = ({ initialValue = "Hello from SampleTest!" }) => {
  const [count, setCount] = useState(0);

  return (
    <div style={{ border: "1px solid #ddd", padding: "16px", margin: "16px" }}>
      <h2>{initialValue}</h2>
      <p>Current count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
};